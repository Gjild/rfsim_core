# src/rfsim_core/circuit_builder.py
import logging
from collections import ChainMap
from typing import Dict, Any, List, Tuple

from .data_structures import Circuit, Net
from .components.base import ComponentBase, COMPONENT_REGISTRY
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
# --- Definitive Architectural Imports for Error Handling ---
# Import the concrete, catchable base exception and the generic formatter.
from .errors import CircuitBuildError, DiagnosableError, format_diagnostic_report


logger = logging.getLogger(__name__)


class CircuitBuilder:
    """
    Synthesizes a simulation-ready, hierarchical `Circuit` object from a parsed IR tree.

    This class implements the "Correctness by Construction" and "Explicit Contracts"
    mandates by transforming the raw, unlinked `ParsedCircuitNode` IR from the
    parser into a final, linked, and simulation-ready `Circuit` model.

    The process is a strict, two-pass pipeline to correctly handle hierarchical
    parameter scoping and overrides:

    1.  **Definition Collection Pass (`_collect_and_scope_definitions`):**
        Recursively traverses the entire IR tree to discover all parameter definitions,
        handle hierarchical overrides, and build the lexical scope maps (`ChainMap`)
        for every parameter. This pass produces a flat list of all final
        `ParameterDefinition` objects required by the ParameterManager.

    2.  **Model Synthesis Pass (`_synthesize_circuit_object_tree`):**
        With the parameters fully defined, this pass recursively traverses the IR
        tree again to instantiate the final `Circuit` and `ComponentBase` simulation
        objects, linking them to a single global `ParameterManager` and performing
        ground net unification.
    """

    def build_simulation_model(self, parsed_tree_root: ParsedCircuitNode) -> Circuit:
        """
        The main build-time entry point. Synthesizes the IR tree into a simulation model.
        This method implements the top-level error handling for the entire build process.

        Args:
            parsed_tree_root: The root of the `ParsedCircuitNode` tree from the parser.

        Returns:
            The fully synthesized, simulation-ready, top-level `Circuit` object.

        Raises:
            CircuitBuildError: A user-facing, diagnosable error if any part of the
                               build process fails. The original exception is chained.
        """
        logger.info(f"--- Starting circuit model synthesis for '{parsed_tree_root.circuit_name}' ---")
        try:
            # --- PASS 1: Collect all parameter definitions and build scopes ---
            logger.debug("Starting Pass 1: Parameter Definition and Scope Collection...")
            all_definitions, scope_maps = self._collect_and_scope_definitions(parsed_tree_root)
            logger.debug("Pass 1 Complete. Collected %d final parameter definitions.", len(all_definitions))

            # --- Instantiate and build the single, global ParameterManager ---
            global_pm = ParameterManager()
            global_pm.build(all_definitions, scope_maps)

            # --- PASS 2: Synthesize the final simulation object tree ---
            logger.debug("Starting Pass 2: Simulation Object Tree Synthesis...")
            sim_circuit = self._synthesize_circuit_object_tree(parsed_tree_root, global_pm)
            logger.debug("Pass 2 Complete. Final circuit object tree synthesized.")

            logger.info(f"--- Circuit model synthesis for '{sim_circuit.name}' successful. ---")
            return sim_circuit

        # ==============================================================================
        # === DEFINITIVE ERROR HANDLING BLOCK ==========================================
        # ==============================================================================
        except DiagnosableError as e:
            # We catch our custom, concrete base class for all reportable application
            # failures (parsing, validation, parameter errors, etc.). We know `e`
            # has the .get_diagnostic_report() method because it's enforced by the
            # DiagnosableError abstract base class contract.
            diagnostic_report = e.get_diagnostic_report()
            raise CircuitBuildError(diagnostic_report) from e

        except Exception as e:
            # This is the fallback for truly unexpected issues (bugs, environment
            # problems, third-party library errors). It ensures the simulator
            # fails gracefully with a helpful message instead of crashing.
            report = format_diagnostic_report(
                error_type=f"An Unexpected Error Occurred ({type(e).__name__})",
                details=f"The circuit builder encountered an unexpected internal error: {str(e)}",
                suggestion="This may indicate a bug in RFSim Core. Please review the traceback, check for input errors, and if the problem persists, consider filing a bug report.",
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

        This method walks the IR tree, creating `ParameterDefinition` objects and the
        `ChainMap` lexical scopes needed for the `ExpressionPreprocessor`. It correctly
        handles subcircuit parameter overrides by replacing definitions from the
        sub-tree with new ones.

        Args:
            ir_node: The current `ParsedCircuitNode` being processed.
            parent_scope: The lexical scope of the parent circuit.
            parent_fqn: The fully qualified name of the parent context.

        Returns:
            A tuple containing:
            - A flat list of all final `ParameterDefinition` objects in this sub-tree.
            - A dictionary mapping every parameter FQN to its correct `ChainMap` scope.
        """
        all_definitions: List[ParameterDefinition] = []
        all_scope_maps: Dict[str, ChainMap] = {}

        # The local scope map contains FQNs for parameters and components defined at this level.
        local_scope_map: Dict[str, str] = {}
        current_scope = parent_scope.new_child(local_scope_map)

        # Process top-level parameters in the current circuit definition file
        for name, value_info in ir_node.raw_parameters_dict.items():

            # --- START OF THE FIX ---
            # Correctly handle the two allowed formats for a parameter value.
            if isinstance(value_info, dict):
                # Format is: {expression: "...", dimension: "..."}
                expression_str = str(value_info.get('expression'))
                dimension_str = str(value_info.get('dimension', 'dimensionless'))
            else:
                # Format is a simple string or number. Assume dimensionless by convention
                # for these top-level "interface" parameters.
                expression_str = str(value_info)
                dimension_str = "dimensionless"
            # --- END OF THE FIX ---

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
                    # This will be caught by SemanticValidator. We skip here to prevent a crash.
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
                        # Add deep name (e.g., 'R1.resistance') to the local scope
                        local_scope_map[f"{comp_ir.instance_id}.{param_name}"] = param_def.fqn
                        all_scope_maps[param_def.fqn] = current_scope

            elif isinstance(comp_ir, ParsedSubcircuitData):
                # Recursively process the subcircuit's definition first
                sub_definitions, sub_scope_maps = self._collect_and_scope_definitions(
                    ir_node=comp_ir.sub_circuit_definition_node,
                    parent_scope=current_scope,
                    parent_fqn=component_fqn
                )

                # Create a lookup map from a parameter's base name (or deep name) within the subcircuit
                # to its full definition object. This is crucial for finding the correct
                # `declared_dimension_str` when applying an override.
                sub_def_lookup: Dict[str, ParameterDefinition] = {
                    p.fqn.replace(f"{component_fqn}.", "", 1): p for p in sub_definitions
                }

                # Apply overrides from the instance definition
                for override_key, override_value in comp_ir.raw_parameter_overrides.items():
                    if override_key not in sub_def_lookup:
                        raise ParameterDefinitionError(
                            fqn=f"{component_fqn}(override)",
                            user_input=override_key,
                            source_yaml_path=comp_ir.source_yaml_path,
                            details=f"Subcircuit instance '{comp_ir.instance_id}' attempts to override parameter '{override_key}', which does not exist in its definition ('{comp_ir.definition_file_path.name}')."
                        )

                    original_def = sub_def_lookup[override_key]

                    # Create the new definition for the override. It uses the new value but
                    # crucially inherits the declared dimension from the original definition.
                    override_def = ParameterDefinition(
                        owner_fqn=original_def.owner_fqn,
                        base_name=original_def.base_name,
                        raw_value_or_expression_str=str(override_value),
                        source_yaml_path=comp_ir.source_yaml_path, # Source is the overriding file
                        declared_dimension_str=original_def.declared_dimension_str
                    )

                    # Replace the original definition with the override.
                    sub_definitions = [d for d in sub_definitions if d.fqn != original_def.fqn]
                    sub_definitions.append(override_def)

                # Add the final, potentially overridden, sub-definitions and their scopes.
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

        This method walks the IR tree a second time, creating the final `Circuit`, `Net`,
        and `ComponentBase` (including `SubcircuitInstance`) objects. It ensures that
        all objects are correctly linked and that there is only one canonical ground `Net`
        object for the entire simulation.

        Args:
            ir_node: The current `ParsedCircuitNode` being processed.
            global_pm: The single, fully-built global ParameterManager.
            parent_fqn: The FQN of the parent context.
            ground_unification_map: A dict passed by reference to ensure a single
                                    canonical ground `Net` object is used everywhere.

        Returns:
            The synthesized `Circuit` object for the current level of the hierarchy.
        """
        if ground_unification_map is None:
            ground_unification_map = {}

        # --- Synthesize Nets for this circuit level ---
        nets: Dict[str, Net] = {}
        raw_net_names = {comp.raw_ports_dict[p] for comp in ir_node.components if isinstance(comp, ParsedLeafComponentData) for p in comp.raw_ports_dict}
        raw_net_names.update({p_map for comp in ir_node.components if isinstance(comp, ParsedSubcircuitData) for p_map in comp.raw_port_mapping.values()})
        raw_net_names.add(ir_node.ground_net_name)
        raw_port_nets = {p['id'] for p in ir_node.raw_external_ports_list}
        raw_net_names.update(raw_port_nets)

        for net_name in sorted(list(raw_net_names)):
            is_ground = (net_name == ir_node.ground_net_name)
            if is_ground:
                if "canonical_ground" not in ground_unification_map:
                    # First time we've seen a ground net; create and store it.
                    ground_unification_map["canonical_ground"] = Net(name=ir_node.ground_net_name, is_ground=True)
                # All nets named 'gnd' (or the specified ground name) point to the same object.
                nets[net_name] = ground_unification_map["canonical_ground"]
            else:
                nets[net_name] = Net(name=net_name, is_external=(net_name in raw_port_nets))

        # --- Synthesize Components for this circuit level ---
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
                # Recursively synthesize the subcircuit's definition first.
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

        # --- Synthesize the final Circuit object for this level ---
        external_ports = {p['id']: nets[p['id']] for p in ir_node.raw_external_ports_list}

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