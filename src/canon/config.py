import logging
import sys

def setup_logger(name: str = "ReconstructionPipeline", level=logging.INFO):
    """Configura logger com saída no console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler de console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formato das mensagens
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger