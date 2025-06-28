# src/rfsim_core/validation/semantic_validator.py
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
import pint

from ..data_structures import Circuit
from ..units import ureg, Quantity
from .issues import ValidationIssue, ValidationIssueLevel
from .issue_codes import SemanticIssueCode
from ..components import COMPONENT_REGISTRY, ComponentBase, Resistor, Capacitor, Inductor
from ..components.subcircuit import SubcircuitInstance
from ..parameters import ParameterManager, ParameterScopeError, ParameterError
from ..parser.raw_data import ParsedLeafComponentData, ParsedSubcircuitData


logger = logging.getLogger(__name__)


class SemanticValidationError(ValueError):
    """
    Custom exception raised when semantic validation detects one or more errors.
    Contains a list of all error-level ValidationIssue objects.
    """
    def __init__(self, issues: List[ValidationIssue]):
        self.issues: List[ValidationIssue] = [
            issue for issue in issues if issue.level == ValidationIssueLevel.ERROR
        ]
        error_lines = [str(issue) for issue in self.issues]
        summary_message = (
            f"Semantic validation failed with {len(self.issues)} error(s):\n"
            + "\n".join(f"  - {line}" for line in error_lines)
        )
        super().__init__(summary_message)


class SemanticValidator:
    """
    Performs recursive semantic validation on a synthesized, hierarchical circuit model.
    """

    def __init__(self, top_level_circuit: Circuit):
        if not isinstance(top_level_circuit, Circuit):
            raise TypeError("SemanticValidator requires a valid top-level Circuit object.")
        if not top_level_circuit.parameter_manager:
            raise ValueError("Top-level Circuit object has an uninitialized ParameterManager.")

        self.top_level_circuit = top_level_circuit
        self.issues: List[ValidationIssue] = []
        self._ureg = ureg
        self.pm: ParameterManager = top_level_circuit.parameter_manager
        self._validated_circuit_defs: Set[str] = set()

    def validate(self) -> List[ValidationIssue]:
        """Public entry point to start the recursive validation process."""
        self.issues = []
        self._validated_circuit_defs.clear()
        logger.info(f"Starting recursive semantic validation for '{self.top_level_circuit.name}'...")
        self._validate_recursive(self.top_level_circuit)

        if self.issues:
            errors = sum(1 for i in self.issues if i.level == ValidationIssueLevel.ERROR)
            warnings = sum(1 for i in self.issues if i.level == ValidationIssueLevel.WARNING)
            infos = sum(1 for i in self.issues if i.level == ValidationIssueLevel.INFO)
            logger.info(f"Validation complete. Found: {errors} errors, {warnings} warnings, {infos} info messages.")
        else:
            logger.info("Validation complete with no issues found.")
            
        return self.issues

    def _add_issue(self, level: ValidationIssueLevel, code_enum: SemanticIssueCode, **kwargs):
        """Helper to create and add a ValidationIssue with full context."""
        comp_fqn = kwargs.get('component_fqn', kwargs.get('instance_fqn'))
        h_context = kwargs.get('hierarchical_context')
        
        if comp_fqn and 'component_fqn' not in kwargs:
             kwargs['component_fqn'] = comp_fqn

        message = code_enum.format_message(**kwargs)
        self.issues.append(ValidationIssue(
            level=level, code=code_enum.code, message=message,
            component_fqn=comp_fqn, hierarchical_context=h_context, details=kwargs
        ))

    def _validate_recursive(self, circuit_node: Circuit):
        """The core recursive validation function."""
        def_path_str = str(circuit_node.source_file_path)
        if def_path_str not in self._validated_circuit_defs:
            logger.debug(f"Validating context for '{circuit_node.hierarchical_id}' defined in '{def_path_str}'")
            self._check_nets_in_context(circuit_node)
            self._check_external_ports_in_context(circuit_node)
            self._validated_circuit_defs.add(def_path_str)

        for sim_comp in circuit_node.sim_components.values():
            if isinstance(sim_comp, SubcircuitInstance):
                self._check_subcircuit_instance(sim_comp)
                self._validate_recursive(sim_comp.sub_circuit_object)
            elif isinstance(sim_comp, ComponentBase):
                self._check_leaf_component_instance(sim_comp)
            else:
                logger.warning(f"Object '{sim_comp.instance_id}' in sim_components is not a ComponentBase; skipping validation.")

    # --- Helper Methods ---
    
    def _get_net_connection_counts(self, circuit_node: Circuit) -> Dict[str, List[Tuple[str, str]]]:
        """
        Helper to compute net connection lists for a specific circuit context.
        Returns a map from net_name -> list of (component_fqn, port_id) tuples.
        """
        connections: Dict[str, List[Tuple[str, str]]] = {}
        
        for sim_comp in circuit_node.sim_components.values():
            ports_map = {}
            if hasattr(sim_comp, 'raw_ir_data'):
                if isinstance(sim_comp, SubcircuitInstance):
                    ports_map = sim_comp.raw_ir_data.raw_port_mapping
                elif isinstance(sim_comp.raw_ir_data, ParsedLeafComponentData):
                    ports_map = sim_comp.raw_ir_data.raw_ports_dict
            
            for port_name, connected_net in ports_map.items():
                if connected_net in circuit_node.nets:
                    if connected_net not in connections:
                        connections[connected_net] = []
                    connections[connected_net].append((sim_comp.fqn, str(port_name)))
        return connections

    # --- Context-Level Checks (Run once per circuit definition) ---
    
    def _check_nets_in_context(self, circuit_node: Circuit):
        """Validates nets for a given circuit definition."""
        net_connections = self._get_net_connection_counts(circuit_node)
        internal_nets = set(circuit_node.nets.keys()) - set(circuit_node.external_ports.keys()) - {circuit_node.ground_net_name}

        for net_name in internal_nets:
            connections = net_connections.get(net_name, [])
            count = len(connections)
            if count == 0:
                self._add_issue(ValidationIssueLevel.WARNING, SemanticIssueCode.NET_CONN_001,
                                net_name=net_name, hierarchical_context=circuit_node.hierarchical_id)
            elif count == 1:
                comp_fqn, port = connections[0]
                self._add_issue(ValidationIssueLevel.WARNING, SemanticIssueCode.NET_CONN_002,
                                net_name=net_name, hierarchical_context=circuit_node.hierarchical_id,
                                connected_to_component=comp_fqn, connected_to_port=port)
        
        if circuit_node.sim_components and len(net_connections.get(circuit_node.ground_net_name, [])) == 0:
            self._add_issue(ValidationIssueLevel.WARNING, SemanticIssueCode.GND_CONN_001,
                            net_name=circuit_node.ground_net_name, hierarchical_context=circuit_node.hierarchical_id)

    def _check_external_ports_in_context(self, circuit_node: Circuit):
        """Validates the external port definitions of a circuit definition, including Z0."""
        net_connections = self._get_net_connection_counts(circuit_node)
        raw_ports_info = {p['id']: p for p in circuit_node.raw_ir_root.raw_external_ports_list}

        for port_name in circuit_node.external_ports:
            if port_name == circuit_node.ground_net_name:
                self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.EXT_PORT_001,
                                net_name=port_name, hierarchical_context=circuit_node.hierarchical_id)
            if len(net_connections.get(port_name, [])) == 0:
                self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.EXT_PORT_002,
                                net_name=port_name, hierarchical_context=circuit_node.hierarchical_id)

            z0_str = raw_ports_info.get(port_name, {}).get('reference_impedance')
            if not z0_str:
                self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.EXT_PORT_Z0_MISSING,
                                net_name=port_name, hierarchical_context=circuit_node.hierarchical_id)
                continue
            
            try:
                qty = self._ureg.Quantity(z0_str)
                if not qty.is_compatible_with("ohm"):
                    self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.EXT_PORT_Z0_DIM_MISMATCH,
                                    net_name=port_name, hierarchical_context=circuit_node.hierarchical_id,
                                    value=z0_str, parsed_dimensionality=str(qty.dimensionality))
            except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError):
                pass # Assume it's a parameter reference; PM build failure is the source of truth for invalid refs.

    # --- Instance-Level Checks ---

    def _check_leaf_component_instance(self, sim_comp: ComponentBase):
        """Validates a single leaf component instance (R, L, C, etc.)."""
        try:
            assert hasattr(sim_comp, 'raw_ir_data') and isinstance(sim_comp.raw_ir_data, ParsedLeafComponentData), \
                   f"Internal Contract Violated: Leaf component '{sim_comp.fqn}' is missing its 'raw_ir_data' link."
        except AssertionError as e:
            raise TypeError(str(e)) from e

        comp_ir = sim_comp.raw_ir_data
        
        if sim_comp.component_type not in COMPONENT_REGISTRY:
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.COMP_TYPE_001,
                            component_fqn=sim_comp.fqn, component_type=sim_comp.component_type,
                            available_types=list(COMPONENT_REGISTRY.keys()))
            return

        declared_ports = {str(p) for p in type(sim_comp).declare_ports()}
        used_ports = {str(p) for p in comp_ir.raw_ports_dict.keys()}
        
        if extra := sorted(list(used_ports - declared_ports)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.COMP_LEAF_PORT_DEF_UNDECLARED,
                            component_fqn=sim_comp.fqn, extra_ports=extra, declared_ports=sorted(list(declared_ports)))
        if missing := sorted(list(declared_ports - used_ports)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.COMP_LEAF_PORT_DEF_MISSING,
                            component_fqn=sim_comp.fqn, missing_ports=missing, declared_ports=sorted(list(declared_ports)))

        declared_params = type(sim_comp).declare_parameters()
        declared_names = set(declared_params.keys())
        provided_names = set(comp_ir.raw_parameters_dict.keys())

        if undeclared := sorted(list(provided_names - declared_names)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.PARAM_LEAF_UNDCL,
                            component_fqn=sim_comp.fqn, parameter_name=undeclared[0], declared_params=sorted(list(declared_names)))
        if missing := sorted(list(declared_names - provided_names)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.PARAM_LEAF_MISSING,
                            component_fqn=sim_comp.fqn, parameter_name=missing[0], declared_params=sorted(list(declared_names)))

        for name, expected_dim in declared_params.items():
            param_fqn = f"{sim_comp.fqn}.{name}"
            if self.pm.is_constant(param_fqn):
                try:
                    const_val = self.pm.get_constant_value(param_fqn)
                    if not const_val.is_compatible_with(expected_dim):
                        self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.PARAM_LEAF_DIM_MISMATCH,
                                        component_fqn=sim_comp.fqn, parameter_name=name,
                                        resolved_value_str=f"{const_val:~P}", expected_dim_str=expected_dim)
                    self._check_and_report_dc_behavior(sim_comp, name, const_val)
                except ParameterError:
                    pass

    def _check_subcircuit_instance(self, sub_inst: SubcircuitInstance):
        """Performs the subcircuit-specific validation checks."""
        sub_def = sub_inst.sub_circuit_object
        instance_ir = sub_inst.raw_ir_data
        instance_fqn = sub_inst.fqn

        declared_sub_ports = set(sub_def.external_ports.keys())
        mapped_sub_ports = set(instance_ir.raw_port_mapping.keys())

        if declared_sub_ports and not instance_ir.raw_port_mapping:
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.SUB_INST_PORT_MAP_REQUIRED,
                            instance_fqn=instance_fqn, sub_def_name=sub_def.name)
            return

        if undeclared := sorted(list(mapped_sub_ports - declared_sub_ports)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.SUB_INST_PORT_MAP_UNDECLARED,
                            instance_fqn=instance_fqn, undeclared_sub_port_name=undeclared[0],
                            sub_def_name=sub_def.name, available_sub_ports=sorted(list(declared_sub_ports)))
        if missing := sorted(list(declared_sub_ports - mapped_sub_ports)):
            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.SUB_INST_PORT_MAP_MISSING,
                            instance_fqn=instance_fqn, sub_def_name=sub_def.name, missing_sub_ports=missing)

        for key, value in instance_ir.raw_parameter_overrides.items():
            target_fqn_in_sub = f"{sub_inst.fqn}.{key}"
            try:
                p_def = self.pm.get_parameter_definition(target_fqn_in_sub)
                if not isinstance(value, dict):
                    try:
                        qty = self._ureg.Quantity(str(value))
                        if not qty.is_compatible_with(p_def.declared_dimension_str):
                            self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.SUB_INST_PARAM_OVERRIDE_DIM_MISMATCH,
                                            instance_fqn=instance_fqn, override_target_in_sub=key,
                                            override_value_str=str(value), provided_dim_str=str(qty.dimensionality),
                                            expected_dim_str=p_def.declared_dimension_str)
                    except (pint.UndefinedUnitError, ValueError):
                        pass
            except ParameterScopeError:
                self._add_issue(ValidationIssueLevel.ERROR, SemanticIssueCode.SUB_INST_PARAM_OVERRIDE_UNDECLARED,
                                instance_fqn=instance_fqn, override_target_in_sub=key)

    def _check_and_report_dc_behavior(self, sim_comp: ComponentBase, param_name: str, value_qty: Quantity):
        """Adds INFO issues for components that are ideal DC shorts or opens."""
        comp_fqn = sim_comp.fqn
        value_str = f"{value_qty:~P}"
        mag = value_qty.magnitude

        if isinstance(sim_comp, Resistor) and param_name == "resistance" and mag == 0:
            self._add_issue(ValidationIssueLevel.INFO, SemanticIssueCode.DC_INFO_SHORT_R0, component_fqn=comp_fqn, value_str=value_str)
        elif isinstance(sim_comp, Inductor) and param_name == "inductance":
            if mag == 0:
                self._add_issue(ValidationIssueLevel.INFO, SemanticIssueCode.DC_INFO_SHORT_L0, component_fqn=comp_fqn, value_str=value_str)
            elif np.isposinf(mag):
                self._add_issue(ValidationIssueLevel.INFO, SemanticIssueCode.DC_INFO_OPEN_LINF, component_fqn=comp_fqn, value_str=value_str)
        elif isinstance(sim_comp, Capacitor) and param_name == "capacitance":
            if mag == 0:
                self._add_issue(ValidationIssueLevel.INFO, SemanticIssueCode.DC_INFO_OPEN_C0, component_fqn=comp_fqn, value_str=value_str)
            elif np.isposinf(mag):
                self._add_issue(ValidationIssueLevel.INFO, SemanticIssueCode.DC_INFO_SHORT_CINF, component_fqn=comp_fqn, value_str=value_str)