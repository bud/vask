"""Slack plugin — Slack Web API tools for vask."""

from vask.plugins.slack.tools import (
    SlackListChannels,
    SlackReact,
    SlackSearchMessages,
    SlackSendMessage,
    SlackSetStatus,
)

TOOLS: list = [
    SlackSendMessage,
    SlackListChannels,
    SlackSearchMessages,
    SlackSetStatus,
    SlackReact,
]
