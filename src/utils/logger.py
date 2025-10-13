import yaml
import logging
import logging.config
from pathlib import Path

def setup_logging(config_path="setup_logging.yaml"):
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning("Logging config file not found; using default INFO level.")
    logger = logging.getLogger(__name__)
    return logger
