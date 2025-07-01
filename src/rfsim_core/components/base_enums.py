# src/rfsim_core/components/base_enums.py
from enum import Enum, auto


class DCBehaviorType(Enum):
    """
    Defines the different ways a component can behave at DC (F=0), which is
    queried by the DCAnalyzer.
    """
    SHORT_CIRCUIT = auto()  # Behaves as a perfect short circuit (e.g., R=0).
    OPEN_CIRCUIT = auto()   # Behaves as a perfect open circuit (e.g., C=finite).
    ADMITTANCE = auto()     # Provides a finite admittance value or N-port matrix.