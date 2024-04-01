# logging_config.py

import logging
import logging.config
import sys

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simpleFormatter": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simpleFormatter",
            "stream": sys.stdout,
        },
        "fileHandler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simpleFormatter",
            "filename": "record.log",
        },
    },
    "loggers": {
        "main": {
            "level": "DEBUG",
            "handlers": ["consoleHandler", "fileHandler"],
        }
    }
}

# 使用配置进行日志设置
logging.config.dictConfig(LOGGING_CONFIG)

# 创建logger对象
logger = logging.getLogger('main')
