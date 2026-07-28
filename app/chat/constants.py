"""Fixed, non-model strings the chat turn path emits."""

from __future__ import annotations

# Sent when a turn arrives on a session with no delivery address chosen yet.
# Swiggy binds a cart to an address, so `update_cart` cannot run without one -
# but that is a precondition of the turn, not something the agent decides, so
# the turn is answered here and the graph is never invoked (no model call is
# spent on a turn that cannot finish). The app resolves it with the address
# picker, which writes `chat_session.selected_address_id`.
ADDRESS_REQUIRED_MESSAGE = (
    "Before I can put a basket together, pick the address you want this "
    "delivered to - tap the address at the top of the chat and choose one."
)
