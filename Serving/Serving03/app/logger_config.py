"""로깅 설정. 비동기 처리 노트 5.3절의 것에 propagate = False를 더했습니다."""
import logging
import sys


def setup_logger(name="ml_api", level="INFO"):
    """콘솔 로거를 만듭니다. 여러 번 불러도 핸들러가 겹치지 않습니다."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
