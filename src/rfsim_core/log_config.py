# --- src/rfsim_core/log_config.py ---
import logging
import sys

def setup_logging(level=logging.INFO):
    """ Configures basic logging to stdout. """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s"
    )
    root_logger = logging.getLogger() # Get the root logger

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    logging.info("Logging configured.")