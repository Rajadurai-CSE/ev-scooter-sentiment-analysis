import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
log_file_path = LOGS_DIR / f"scraper_log_{timestamp}.log"
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
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[file_handler,console_handler]
)

log = logging.getLogger("ev_scraper")