import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

log_file_path = LOGS_DIR / f"application.log"
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    handlers=[file_handler,console_handler]
)

def get_logger(name:str):
    return logging.getLogger(name)
