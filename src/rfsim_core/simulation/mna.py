# src/rfsim_core/simulation/mna.py
import logging
from itertools import product

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Set

from .exceptions import MnaInputError

from ..data_structures import Circuit, Net
from ..parser.raw_data import ParsedLeafComponentData, ParsedSubcircuitData
from ..components.base import ComponentBase, StampInfo
from ..components.exceptions import ComponentError
from ..components.subcircuit import SubcircuitInstance
from ..units import ureg, Quantity
from ..parameters import ParameterManager, ParameterError

logger = logging.getLogger(__name__)


#class MnaInputError(ValueError):
#    """Error related to inputs for MNA assembly."""
#    pass


class MnaAssembler:
    def __init__(self, circuit: Circuit, active_nets_override: Optional[Set[str]] = None):
        if not isinstance(circuit, Circuit) or not circuit.raw_ir_root:
            raise MnaInputError("Circuit object must be simulation-ready.")
        
        self.circuit_orig: Circuit = circuit 
        self.ureg = ureg
        self.parameter_manager: ParameterManager = circuit.parameter_manager

        self.active_nets_override: Optional[Set[str]] = active_nets_override
        
        self._effective_nets: Dict[str, Net] 
        self._effective_sim_components_map: Dict[str, ComponentBase] 
        self._effective_external_ports: Dict[str, Net] 
        self._effective_ground_net_name: Optional[str] = None

        self._filter_circuit_elements() 

        self.node_map: Dict[str, int] = {}
        self.node_count: int = 0
        self.port_names: List[str] = [] 
        self.port_indices: List[int] = [] 

        self._external_node_indices_reduced: List[int] = []
        self._internal_node_indices_reduced: List[int] = []

        self._cached_rows: Optional[np.ndarray] = None
        self._cached_cols: Optional[np.ndarray] = None
        self._sparsity_nnz: int = 0
        self._shape_full: Tuple[int, int] = (0, 0)
        
        self._assign_node_indices()      
        self._identify_reduced_indices()  
        self._compute_sparsity_pattern()  

        logger.info(f"MNA Assembler initialized for circuit '{self.circuit_orig.name}'.")

    @property
    def effective_sim_components(self) -> Dict[str, ComponentBase]:
        return self._effective_sim_components_map
        
    def _filter_circuit_elements(self):
        if self.active_nets_override is None:
            self._effective_nets = self.circuit_orig.nets
            self._effective_sim_components_map = self.circuit_orig.sim_components
            self._effective_external_ports = self.circuit_orig.external_ports
            self._effective_ground_net_name = self.circuit_orig.ground_net_name
            return

        active_nets_set = self.active_nets_override
        self._effective_nets = {n: net for n, net in self.circuit_orig.nets.items() if n in active_nets_set}
        self._effective_ground_net_name = self.circuit_orig.ground_net_name if self.circuit_orig.ground_net_name in active_nets_set else None
        
        self._effective_sim_components_map = {}
        for comp_id, sim_comp in self.circuit_orig.sim_components.items():
            raw_comp = sim_comp.raw_ir_data
            connected_nets = set()
            if isinstance(raw_comp, ParsedLeafComponentData):
                connected_nets.update(raw_comp.raw_ports_dict.values())
            elif isinstance(raw_comp, ParsedSubcircuitData):
                connected_nets.update(raw_comp.raw_port_mapping.values())
            
            if connected_nets.issubset(active_nets_set):
                self._effective_sim_components_map[sim_comp.fqn] = sim_comp
        
        self._effective_external_ports = {n: net for n, net in self.circuit_orig.external_ports.items() if n in active_nets_set}

    def _assign_node_indices(self):
        if not self._effective_nets:
            self.node_count = 0
            self.port_names, self.port_indices = [], []
            self._shape_full = (0, 0)
            return

        self.node_map = {}
        idx = 0
        if self._effective_ground_net_name:
            self.node_map[self._effective_ground_net_name] = idx
            idx += 1
        
        for net_name in sorted(self._effective_nets.keys()):
            if net_name not in self.node_map:
                self.node_map[net_name] = idx
                idx += 1
        
        self.node_count = idx
        self._shape_full = (idx, idx)
        self.port_names = sorted(self._effective_external_ports.keys())
        self.port_indices = [self.node_map[name] for name in self.port_names]

    def _identify_reduced_indices(self):
        if self.node_count <= 1:
            self._external_node_indices_reduced, self._internal_node_indices_reduced = [], []
            return

        ground_is_zero = self._effective_ground_net_name is not None
        all_indices_set = set(range(self.node_count))
        port_indices_set = set(self.port_indices)
        
        if ground_is_zero:
            all_indices_set.remove(0)
            port_indices_set.discard(0)
        
        reduced_map = {full_idx: (full_idx - 1 if ground_is_zero else full_idx) for full_idx in all_indices_set}
        self._external_node_indices_reduced = sorted([reduced_map[i] for i in port_indices_set])
        internal_indices_set = all_indices_set - port_indices_set
        self._internal_node_indices_reduced = sorted([reduced_map[i] for i in internal_indices_set])

    @property
    def external_node_indices_reduced(self) -> List[int]: return self._external_node_indices_reduced
    
    @property
    def internal_node_indices_reduced(self) -> List[int]: return self._internal_node_indices_reduced

    def _add_to_coo_pattern(self, rows: list, cols: list, r: int, c: int):
        rows.append(r); cols.append(c)

    def _compute_sparsity_pattern(self):
        rows, cols = [], []
        if self.node_count == 0:
            self._cached_rows, self._cached_cols, self._sparsity_nnz = np.array([], dtype=int), np.array([], dtype=int), 0
            return

        for sim_comp in self._effective_sim_components_map.values():
            if isinstance(sim_comp, SubcircuitInstance):
                indices = {self.node_map[n] for n in sim_comp.raw_ir_data.raw_port_mapping.values() if n in self.node_map}
                for r, c in product(sorted(list(indices)), repeat=2):
                    self._add_to_coo_pattern(rows, cols, r, c)
            else:
                raw_comp = sim_comp.raw_ir_data
                for p1, p2 in type(sim_comp).declare_connectivity():
                    n1_name = raw_comp.raw_ports_dict.get(p1, raw_comp.raw_ports_dict.get(str(p1)))
                    n2_name = raw_comp.raw_ports_dict.get(p2, raw_comp.raw_ports_dict.get(str(p2)))
                    if n1_name in self.node_map and n2_name in self.node_map:
                        n1, n2 = self.node_map[n1_name], self.node_map[n2_name]
                        self._add_to_coo_pattern(rows, cols, n1, n1)
                        self._add_to_coo_pattern(rows, cols, n2, n2)
                        if n1 != n2:
                            self._add_to_coo_pattern(rows, cols, n1, n2)
                            self._add_to_coo_pattern(rows, cols, n2, n1)

        temp_coo = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=self._shape_full)
        final_coo = temp_coo.tocsc().tocoo()
        self._cached_rows, self._cached_cols, self._sparsity_nnz = final_coo.row, final_coo.col, final_coo.nnz

    def assemble(self, current_sweep_idx: int, all_stamps_vectorized: Dict[str, List[StampInfo]]) -> sp.csc_matrix:
        if self._cached_rows is None: raise RuntimeError("Sparsity pattern not computed.")
        if self.node_count == 0: return sp.csc_matrix((0, 0), dtype=np.complex128)

        mna_data = np.zeros(self._sparsity_nnz, dtype=np.complex128)
        rc_to_data_idx = {(r, c): i for i, (r, c) in enumerate(zip(self._cached_rows, self._cached_cols))}

        def add_value(r, c, value, comp_fqn):
            if (key := rc_to_data_idx.get((r, c))) is not None:
                mna_data[key] += value
            elif abs(value) > 1e-15:
                raise MnaInputError(f"Comp '{comp_fqn}' tried to stamp ({value:.3e}) at ({r},{c}), outside sparsity.")

        for comp_fqn, sim_comp in self._effective_sim_components_map.items():
            try:
                stamp_infos = all_stamps_vectorized[comp_fqn]
                
                for stamp_matrix_qty, port_ids_yaml in stamp_infos:
                    # Slice the correct per-frequency matrix from the pre-computed vectorized stamp
                    stamp_matrix_scalar = stamp_matrix_qty.magnitude[current_sweep_idx]
                    
                    raw_comp = sim_comp.raw_ir_data
                    port_map = raw_comp.raw_port_mapping if isinstance(raw_comp, ParsedSubcircuitData) else raw_comp.raw_ports_dict
                    
                    global_indices = [self.node_map[port_map[pid]] for pid in port_ids_yaml]
                    
                    for i, r_idx in enumerate(global_indices):
                        for j, c_idx in enumerate(global_indices):
                            add_value(r_idx, c_idx, stamp_matrix_scalar[i, j], comp_fqn)

            except Exception as e:
                # FIX: Use keyword arguments to correctly instantiate the dataclass
                raise ComponentError(
                    component_fqn=comp_fqn,
                    details=f"Stamping failed at sweep index {current_sweep_idx}: {e}"
                ) from e

        Yn_full_coo = sp.coo_matrix((mna_data, (self._cached_rows, self._cached_cols)), shape=self._shape_full)
        return Yn_full_coo.tocsc()