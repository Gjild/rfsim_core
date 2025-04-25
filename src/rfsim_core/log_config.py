# src/rfsim_core/log_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configures basic logging to stdout.
    """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s"
    )
    root_logger = logging.getLogger() # Get the root logger

    # Clear existing handlers (useful if this function is called multiple times, e.g., in tests)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Set the level for the root logger
    # Note: Handlers can have their own levels, but setting the root level
    # acts as a primary filter.
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Optionally silence overly verbose libraries if needed later
    # logging.getLogger("some_verbose_library").setLevel(logging.WARNING)

    logging.info("Logging configured.")

# You might want to configure logging immediately when this module is imported,
# or explicitly call setup_logging from your main entry point or __init__.py
# For now, let's just define the function.