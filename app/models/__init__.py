"""ORM models for Cue's Alembic-owned tables.

Import from here rather than the per-domain modules so `Base.metadata` stays
the single source of truth. `app.models_registry` imports this package to
register every table before autogenerate or the test harness runs.
"""

from __future__ import annotations

from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatMessage, ChatSession
from app.models.order import Order
from app.models.pantry import PantryItem
from app.models.provider import OAuthTransaction, ProviderLink
from app.models.user import User

__all__ = [
    "CartPlan",
    "CartPlanItem",
    "ChatMessage",
    "ChatSession",
    "OAuthTransaction",
    "Order",
    "PantryItem",
    "ProviderLink",
    "User",
]
