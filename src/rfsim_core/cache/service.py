# src/rfsim_core/cache/service.py
"""
Provides the centralized, multi-scope caching service for the entire simulation run.
"""
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class SimulationCache:
    """
    A centralized, multi-scope caching service for a simulation run. It manages two
    distinct cache scopes:

    - 'run': An instance-level cache whose lifetime is tied to this specific
             SimulationCache object. It is used for caching results that are only
             valid within a single, top-level `run_sweep` call (e.g., subcircuit sims
             that depend on frequency-varying external parameters).

    - 'process': A class-level (static) cache that persists across multiple
                 `run_sweep` calls within the same Python process. It is used for
                 caching results that are guaranteed to be identical for a given
                 set of inputs, regardless of the run context (e.g., topology or
                 DC analysis, which only depend on constant parameters).

    This design directly addresses the "Hidden Global State" anti-pattern by making
    the cache an explicit, managed service rather than a hidden static variable within
    an analysis tool.
    """
    # This class-level dictionary is the single source of truth for the persistent cache.
    _process_cache: Dict[Tuple, Any] = {}

    def __init__(self):
        # This instance-level dictionary is reset for every new SimulationCache instance.
        self._run_cache: Dict[Tuple, Any] = {}
        self.clear_stats()
        logger.debug("SimulationCache instance created.")

    def get(self, key: Tuple, scope: str = 'run') -> Any:
        """Retrieves an item from the specified cache scope."""
        cache = self._get_cache_for_scope(scope)
        if key in cache:
            self._stats[scope]['hits'] += 1
            # Logging the key can be verbose; truncate for readability.
            logger.debug(f"Cache HIT in '{scope}' scope for key: {str(key)[:150]}...")
            return cache[key]

        self._stats[scope]['misses'] += 1
        logger.debug(f"Cache MISS in '{scope}' scope for key: {str(key)[:150]}...")
        return None

    def put(self, key: Tuple, value: Any, scope: str = 'run'):
        """Stores an item in the specified cache scope."""
        cache = self._get_cache_for_scope(scope)
        if key in cache:
            logger.warning(f"Cache key collision detected in '{scope}' scope. Overwriting existing value.")
        cache[key] = value

    def _get_cache_for_scope(self, scope: str) -> Dict[Tuple, Any]:
        """A private helper to select the correct dictionary for the scope."""
        if scope == 'run':
            return self._run_cache
        if scope == 'process':
            return self.__class__._process_cache
        raise ValueError(f"Invalid cache scope '{scope}'. Must be 'run' or 'process'.")

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Returns a copy of the cache hit/miss statistics for this run."""
        # Create a deep copy to prevent external modification of internal state.
        return {
            'run': self._stats['run'].copy(),
            'process': self._stats['process'].copy()
        }

    def clear_stats(self):
        """Resets the hit/miss statistics for this instance."""
        self._stats = {'run': {'hits': 0, 'misses': 0}, 'process': {'hits': 0, 'misses': 0}}

    @classmethod
    def clear_process_cache(cls):
        """
        A class method to explicitly clear the persistent process-level cache.
        This is crucial for testing and for advanced users who want to force
        re-computation in a long-running interactive session.
        """
        cls._process_cache.clear()
        logger.info("Cleared the persistent process-level cache.")