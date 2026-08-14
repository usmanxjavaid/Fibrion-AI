"""
Centralized logging for Fibrion.

Every agent gets its own named logger (fibrion.agents.<name>). Every log
line is automatically tagged with the current pipeline run_id via a
contextvar, so logs from one upload's run can be filtered from every
other run without threading a run_id argument through every function
call by hand.
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

run_id_ctx: ContextVar[str] = ContextVar("run_id", default="-")


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    env = os.getenv("FIBRION_ENV", "development")
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RunIdFilter())

    if env == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-28s | run=%(run_id)s | %(message)s"
            )
        )

    root = logging.getLogger("fibrion")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def get_agent_logger(agent_name: str) -> logging.Logger:
    return logging.getLogger(f"fibrion.agents.{agent_name}")