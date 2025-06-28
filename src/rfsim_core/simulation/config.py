# src/rfsim_core/simulation/config.py
import logging
import numpy as np
import pint
from typing import Dict, Any

from ..units import ureg

logger = logging.getLogger(__name__)

class ConfigParsingError(ValueError):
    """Custom exception for errors during simulation configuration parsing."""
    pass

def parse_sweep_config(raw_sweep_config: Dict[str, Any]) -> np.ndarray:
    """
    Parses a raw sweep configuration dictionary into a NumPy frequency array.
    """
    if not raw_sweep_config:
        raise ConfigParsingError("Sweep configuration is missing or empty.")
    try:
        sweep_type = raw_sweep_config['type']
        freq_values_hz = np.array([], dtype=float)

        if sweep_type in ['linear', 'log']:
            start_hz = ureg.Quantity(raw_sweep_config['start']).to('Hz').magnitude
            stop_hz = ureg.Quantity(raw_sweep_config['stop']).to('Hz').magnitude
            num_points = int(raw_sweep_config['num_points'])
            
            if stop_hz < start_hz: raise ValueError("Stop frequency cannot be less than start frequency.")

            if sweep_type == 'linear':
                if start_hz < 0: raise ValueError("Linear sweep start frequency must be >= 0.")
                freq_values_hz = np.linspace(start_hz, stop_hz, num_points, dtype=float)
            else: # log
                if start_hz <= 0 or stop_hz <= 0: raise ValueError("Log sweep frequencies must be > 0.")
                freq_values_hz = np.geomspace(start_hz, stop_hz, num_points, dtype=float)

        elif sweep_type == 'list':
            points = [ureg.Quantity(p).to('Hz').magnitude for p in raw_sweep_config['points']]
            if any(f < 0 for f in points): raise ValueError("Frequencies in list must be non-negative.")
            freq_values_hz = np.array(sorted(list(set(points))), dtype=float)
            
        return freq_values_hz
    except (KeyError, ValueError, pint.DimensionalityError, pint.UndefinedUnitError) as e:
        raise ConfigParsingError(f"Failed to parse sweep configuration: {e}") from e