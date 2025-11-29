"""
通用日志模块
日志输出到控制台和文件
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件名：按日期
LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"


def setup_logger(name: str = "deepsearch") -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # 日志格式：简单通用
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# 创建全局日志器
logger = setup_logger()


def log(message: str, level: str = "info"):
    """
    简单的日志函数
    
    Args:
        message: 日志消息
        level: 日志级别 (debug/info/warning/error)
    """
    level = level.lower()
    if level == "debug":
        logger.debug(message)
    elif level == "warning" or level == "warn":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)


# 便捷方法
def info(message: str):
    logger.info(message)

def debug(message: str):
    logger.debug(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)


if __name__ == "__main__":
    # 测试日志
    print(f"日志文件: {LOG_FILE}")
    print()
    
    log("这是一条普通日志")
    log("这是一条调试日志", "debug")
    log("这是一条警告日志", "warning")
    log("这是一条错误日志", "error")
    
    print()
    print(f"日志已写入: {LOG_FILE}")

