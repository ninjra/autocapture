"""Hotness overlay tracker module."""

# Conversation state guide (why UI scraping is out of scope):
# https://platform.openai.com/docs/guides/conversation-state

from .service import OverlayTrackerService

__all__ = ["OverlayTrackerService"]
