"""Server-sent-events framing for the chat stream.

One place that knows the wire format, so the service can yield typed events
and stay unaware of how they are encoded.
"""

from __future__ import annotations

from app.chat.schemas import ChatStreamEvent

#: Sent to keep intermediaries from closing an idle connection. A comment
#: line is valid SSE and is ignored by every client, so it costs the app
#: nothing to receive.
KEEPALIVE = ": keepalive\n\n"


def format_event(event: ChatStreamEvent) -> str:
    """Encode one typed event as an SSE frame.

    Emitted as a *named* event (`event:` line) so a client dispatches on the
    name rather than sniffing the payload's shape.

    Args:
        event: The event to send.

    Returns:
        The complete frame, terminated by the blank line SSE requires.
    """
    return f"event: {event.event.value}\ndata: {event.model_dump_json()}\n\n"
