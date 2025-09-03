"""
Project Maestro: AI Agent-based Game Prototyping Automation System

A sophisticated multi-agent orchestration system that converts game design 
documents into playable game prototypes using LangChain and specialized AI agents.
"""

__version__ = "0.1.0"
__author__ = "Project Maestro Team"
__email__ = "team@projectmaestro.dev"

from .core.config import settings
from .core.logging import logger

__all__ = ["settings", "logger"]