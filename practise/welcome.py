import logging

def get_logger(name):
    # Handlers
    info_handler = logging.FileHandler("info.log")
    error_handler = logging.FileHandler("error.log")
    debug_handler = logging.FileHandler("debug.log")

    # Levels
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)
    debug_handler.setLevel(logging.DEBUG)

    # Format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate logs (important in reusability)
    if not logger.handlers:
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        logger.addHandler(debug_handler)

    return logger
