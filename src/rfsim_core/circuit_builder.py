# src/rfsim_core/circuit_builder.py

"""
Defines the CircuitBuilder, a critical component of the simulation core responsible
for synthesizing a simulation-ready, hierarchical `Circuit` object from a parsed
Intermediate Representation (IR) tree.

Architectural Role:
The CircuitBuilder acts as the bridge between the raw, validated data from the
NetlistParser and the final, interconnected object graph used by the simulation
engine. Its primary responsibility is to perform the complex tasks of:

1.  **Hierarchical Scope Management:** Correctly building and managing lexical
    scopes for parameter resolution across multiple levels of subcircuits using
    `collections.ChainMap`.

2.  **Parameter Definition and Validation:** Collecting all parameter definitions
    from the entire circuit hierarchy, creating formal `ParameterDefinition`
    objects with full context (FQN, source file, declared dimension), and
    passing them to the `ParameterManager`.

3.  **Object Graph Synthesis:** Instantiating the final `Circuit`, `ComponentBase`,
    and `SubcircuitInstance` objects, and correctly linking them together. This
    includes the critical process of **Ground Unification**, where all nets
    named "gnd" are mapped to a single, canonical `Net` object.

4.  **Top-Level Error Handling:** Acting as the primary gatekeeper for build-time
    errors. It wraps its entire process in a robust error handler that catches
    any `DiagnosableError` from its subsystems (Parser, ParameterManager) and
    re-raises it as a single, user-friendly `CircuitBuildError`, ensuring
    actionable diagnostics are always presented to the user.
"""

import logging
from collections import ChainMap
from typing import Dict, Any, List, Tuple

from .data_structures import Circuit, Net
from .components.base import COMPONENT_REGISTRY, ComponentBase
from .components.subcircuit import SubcircuitInstance
from .parser.raw_data import (
    ParsedCircuitNode,
    ParsedLeafComponentData,
    ParsedSubcircuitData,
)
from .parameters import (
    ParameterManager,
    ParameterDefinition,
    ParameterDefinitionError,
)
# This import is critical for the top-level error handling facade.
from .errors import CircuitBuildError, DiagnosableError, format_diagnostic_report


logger = logging.getLogger(__name__)


class CircuitBuilder:
    """
    Synthesizes a simulation-ready, hierarchical `Circuit` object from a parsed IR tree.
    This class implements a two-pass pipeline to handle complex parameter dependencies
    before instantiating the final simulation objects.
    """

    def build_simulation_model(self, parsed_tree_root: ParsedCircuitNode) -> Circuit:
        """
        The main build-time entry point. Synthesizes the IR tree into a simulation model.
        This method orchestrates the two-pass build process and provides the definitive
        top-level error handling for the entire build stage.
        """
        logger.info(f"--- Starting circuit model synthesis for '{parsed_tree_root.circuit_name}' ---")
        try:
            logger.debug("Starting Pass 1: Parameter Definition and Scope Collection...")
            all_definitions, scope_maps = self._collect_and_scope_definitions(parsed_tree_root)
            logger.debug("Pass 1 Complete. Collected %d final parameter definitions.", len(all_definitions))

            global_pm = ParameterManager()
            global_pm.build(all_definitions, scope_maps)

            logger.debug("Starting Pass 2: Simulation Object Tree Synthesis...")
            sim_circuit = self._synthesize_circuit_object_tree(parsed_tree_root, global_pm)
            logger.debug("Pass 2 Complete. Final circuit object tree synthesized.")

            logger.info(f"--- Circuit model synthesis for '{sim_circuit.name}' successful. ---")
            return sim_circuit

        except DiagnosableError as e:
            # This is the primary error handling path. It catches any well-defined,
            # diagnosable error from the subsystems (parsing, parameter validation)
            # and formats it into the final user-facing report.
            diagnostic_report = e.get_diagnostic_report()
            raise CircuitBuildError(diagnostic_report) from e
        
        # DEFINITIVE FIX: Harden the generic exception handler.
        except Exception as e:
            # This is a defensive catch-all for unexpected errors. It now checks if the
            # error is a raw SyntaxError that may have escaped a subsystem and provides
            # a more specific report if so.
            if isinstance(e, SyntaxError):
                details = f"The circuit builder encountered an unhandled syntax error: {str(e)}"
                suggestion = "This may indicate a bug in RFSim Core's expression handling. Please check the parameter expressions in your netlists for valid Python syntax."
                error_type = "Unhandled Syntax Error"
            else:
                details = f"The circuit builder encountered an unexpected internal error: {str(e)}"
                suggestion = "This may indicate a bug in RFSim Core. Please review the traceback."
                error_type = f"An Unexpected Error Occurred ({type(e).__name__})"

            report = format_diagnostic_report(
                error_type=error_type,
                details=details,
                suggestion=suggestion,
                context={}
            )
            raise CircuitBuildError(report) from e

    def _collect_and_scope_definitions(
        self,
        ir_node: ParsedCircuitNode,
        parent_scope: ChainMap = ChainMap(),
        parent_fqn: str = "top"
    ) -> Tuple[List[ParameterDefinition], Dict[str, ChainMap]]:
        """
        Recursively performs Pass 1: collecting parameter definitions and building scopes.
        This is the heart of the hierarchical parameter resolution system.
        """
        all_definitions: List[ParameterDefinition] = []
        all_scope_maps: Dict[str, ChainMap] = {}
        local_scope_map: Dict[str, str] = {}
        current_scope = parent_scope.new_child(local_scope_map)

        # Process top-level parameters in the current circuit definition file
        for name, value_info in ir_node.raw_parameters_dict.items():
            # This correctly handles the two allowed formats for a parameter value.
            if isinstance(value_info, dict):
                # Format is: {expression: "...", dimension: "..."}
                # This is the EXPLICIT and CORRECT way to define a parameter with a dimension.
                expression_str = str(value_info.get('expression'))
                dimension_str = str(value_info.get('dimension', 'dimensionless'))
            else:
                # Format is a simple key: value (e.g., gain: 2.0 or gain: other_param).
                # This format is ONLY for dimensionless literals or expressions that
                # are known to result in a dimensionless quantity. This is a non-negotiable contract.
                expression_str = str(value_info)
                dimension_str = "dimensionless"

            param_def = ParameterDefinition(
                owner_fqn=parent_fqn,
                base_name=name,
                raw_value_or_expression_str=expression_str,
                source_yaml_path=ir_node.source_yaml_path,
                declared_dimension_str=dimension_str
            )
            all_definitions.append(param_def)
            local_scope_map[name] = param_def.fqn
            all_scope_maps[param_def.fqn] = current_scope

        # Process all components defined in this file
        for comp_ir in ir_node.components:
            component_fqn = f"{parent_fqn}.{comp_ir.instance_id}"

            if isinstance(comp_ir, ParsedLeafComponentData):
                if comp_ir.component_type not in COMPONENT_REGISTRY:
                    logger.warning(f"Skipping parameter collection for '{component_fqn}' due to unknown type '{comp_ir.component_type}'.")
                    continue

                ComponentClass = COMPONENT_REGISTRY[comp_ir.component_type]
                declared_params = ComponentClass.declare_parameters()

                for param_name, expected_dim in declared_params.items():
                    if param_name in comp_ir.raw_parameters_dict:
                        param_value = comp_ir.raw_parameters_dict[param_name]
                        param_def = ParameterDefinition(
                            owner_fqn=component_fqn,
                            base_name=param_name,
                            raw_value_or_expression_str=str(param_value),
                            source_yaml_path=comp_ir.source_yaml_path,
                            declared_dimension_str=expected_dim
                        )
                        all_definitions.append(param_def)
                        local_scope_map[f"{comp_ir.instance_id}.{param_name}"] = param_def.fqn
                        all_scope_maps[param_def.fqn] = current_scope

            elif isinstance(comp_ir, ParsedSubcircuitData):
                sub_definitions, sub_scope_maps = self._collect_and_scope_definitions(
                    ir_node=comp_ir.sub_circuit_definition_node,
                    parent_scope=current_scope,
                    parent_fqn=component_fqn
                )

                sub_def_lookup: Dict[str, ParameterDefinition] = {
                    p.fqn.replace(f"{component_fqn}.", "", 1): p for p in sub_definitions
                }

                # Apply overrides from the subcircuit instance
                for override_key, override_value in comp_ir.raw_parameter_overrides.items():
                    if override_key not in sub_def_lookup:
                        raise ParameterDefinitionError(
                            fqn=f"{component_fqn}(override)",
                            user_input=override_key,
                            source_yaml_path=comp_ir.source_yaml_path,
                            details=f"Subcircuit instance '{comp_ir.instance_id}' attempts to override parameter '{override_key}', which does not exist in its definition ('{comp_ir.definition_file_path.name}')."
                        )

                    original_def = sub_def_lookup[override_key]
                    override_def = ParameterDefinition(
                        owner_fqn=original_def.owner_fqn,
                        base_name=original_def.base_name,
                        raw_value_or_expression_str=str(override_value),
                        source_yaml_path=comp_ir.source_yaml_path,
                        # CRITICAL: The declared dimension is sourced from the subcircuit's
                        # original definition, not the override value.
                        declared_dimension_str=original_def.declared_dimension_str
                    )
                    # Replace the original definition with the override
                    sub_definitions = [d for d in sub_definitions if d.fqn != original_def.fqn]
                    sub_definitions.append(override_def)

                # This propagates sub-parameter names up to the parent scope
                # so that they are visible to expressions in the parent circuit.
                for sub_p_def in sub_definitions:
                    # e.g., create a key like 'X1.R_load.resistance' in the parent's scope
                    relative_name = sub_p_def.fqn.replace(f"{parent_fqn}.", "", 1)
                    local_scope_map[relative_name] = sub_p_def.fqn

                all_definitions.extend(sub_definitions)
                all_scope_maps.update(sub_scope_maps)

        return all_definitions, all_scope_maps

    def _synthesize_circuit_object_tree(
        self,
        ir_node: ParsedCircuitNode,
        global_pm: ParameterManager,
        parent_fqn: str = "top",
        ground_unification_map: Dict[str, Net] = None
    ) -> Circuit:
        """
        Recursively performs Pass 2: instantiating all simulation-ready objects.
        This includes the critical Ground Unification process.
        """
        if ground_unification_map is None:
            # The ground unification map is created once at the top level and passed
            # down through the recursion to ensure all grounds map to one object.
            ground_unification_map = {}

        # Collect all unique net names declared in this circuit context
        nets: Dict[str, Net] = {}
        raw_net_names = {comp.raw_ports_dict[p] for comp in ir_node.components if isinstance(comp, ParsedLeafComponentData) for p in comp.raw_ports_dict}
        raw_net_names.update({p_map for comp in ir_node.components if isinstance(comp, ParsedSubcircuitData) for p_map in comp.raw_port_mapping.values()})
        raw_net_names.add(ir_node.ground_net_name)
        raw_port_nets = {p['id'] for p in ir_node.raw_external_ports_list}
        raw_net_names.update(raw_port_nets)

        # Instantiate Net objects, unifying the ground net
        for net_name in sorted(list(raw_net_names)):
            is_ground = (net_name == ir_node.ground_net_name)
            if is_ground:
                if "canonical_ground" not in ground_unification_map:
                    ground_unification_map["canonical_ground"] = Net(name=ir_node.ground_net_name, is_ground=True)
                nets[net_name] = ground_unification_map["canonical_ground"]
            else:
                nets[net_name] = Net(name=net_name, is_external=(net_name in raw_port_nets))

        # Instantiate simulation-ready Component objects
        sim_components: Dict[str, ComponentBase] = {}
        for comp_ir in ir_node.components:
            component_fqn = f"{parent_fqn}.{comp_ir.instance_id}"
            if isinstance(comp_ir, ParsedLeafComponentData):
                if comp_ir.component_type in COMPONENT_REGISTRY:
                    ComponentClass = COMPONENT_REGISTRY[comp_ir.component_type]
                    sim_components[comp_ir.instance_id] = ComponentClass(
                        instance_id=comp_ir.instance_id,
                        parameter_manager=global_pm,
                        parent_hierarchical_id=parent_fqn,
                        raw_ir_data=comp_ir
                    )
            elif isinstance(comp_ir, ParsedSubcircuitData):
                sub_circuit_obj = self._synthesize_circuit_object_tree(
                    ir_node=comp_ir.sub_circuit_definition_node,
                    global_pm=global_pm,
                    parent_fqn=component_fqn,
                    ground_unification_map=ground_unification_map
                )
                sim_components[comp_ir.instance_id] = SubcircuitInstance(
                    instance_id=comp_ir.instance_id,
                    parameter_manager=global_pm,
                    sub_circuit_object_ref=sub_circuit_obj,
                    sub_circuit_external_port_names_ordered=sorted(list(sub_circuit_obj.external_ports.keys())),
                    parent_hierarchical_id=parent_fqn,
                    raw_ir_data=comp_ir
                )

        external_ports = {p['id']: nets[p['id']] for p in ir_node.raw_external_ports_list}

        # Assemble the final, simulation-ready Circuit object for this level
        circuit_obj = Circuit(
            name=ir_node.circuit_name,
            hierarchical_id=parent_fqn,
            source_file_path=ir_node.source_yaml_path,
            ground_net_name=ir_node.ground_net_name,
            nets=nets,
            sim_components=sim_components,
            external_ports=external_ports,
            parameter_manager=global_pm,
            raw_ir_root=ir_node
        )
        return circuit_obj