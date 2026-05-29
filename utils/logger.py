"""
utils/logger.py  —  Centralized Logger  v3.0
============================================
Provides:
  • Rotating file handler (10 MB × 5 backups)
  • Crash dump on unhandled exception (writes traceback to logs/crash_<ts>.txt)
  • Debug mode via env DEBUG=1
  • Single call: setup_logging() in main_app.py / server/main.py
"""

import logging
import logging.handlers
import os
import sys
import traceback
import time

LOG_DIR   = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
LOG_LEVEL = logging.DEBUG if os.environ.get("DEBUG", "0") == "1" else logging.INFO


def setup_logging(app_name: str = "proctoring") -> logging.Logger:
    """Call once at startup. Returns root-like named logger."""
    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-24s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, f"{app_name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    # Console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    # Avoid duplicate handlers on hot-reload
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

    # Crash dump hook
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        dump_path = os.path.join(LOG_DIR, f"crash_{int(time.time())}.txt")
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        except Exception:
            pass
        logging.getLogger(app_name).critical(
            "UNHANDLED EXCEPTION — crash dump: %s\n%s",
            dump_path,
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _excepthook
    return logging.getLogger(app_name)
