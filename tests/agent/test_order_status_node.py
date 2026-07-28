"""`order_status_node`: which order it answers about, and what it refuses to say.

Unit tests - no real model and no real network. The model seam is stubbed the
way every other node test stubs it, and the throttled order read is stubbed so
these cases are about the node rather than about Swiggy.

The poll floor itself is not tested here; it belongs to
`app.orders.service.list_orders_throttled` and is tested in
`tests/orders/test_service.py`.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agent.config import ModelRole
from app.agent.context import CueContext
from app.agent.nodes import order_status as node_module
from app.agent.nodes.order_status import NO_ORDERS_MESSAGE, order_status_node
from app.agent.state import AgentState
from app.instamart.exceptions import InstamartAuthError
from app.orders import service as orders_service
from app.orders.schemas import OrderListItem, OrderStatus


class _FakeChatModel:
    """Stands in for the prose model the node invokes directly."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.prompts: list[Any] = []
        self.calls = 0

    async def ainvoke(self, prompt: list[Any]) -> Any:
        self.calls += 1
        self.prompts.append(prompt)
        return self._replies.pop(0)


def _state() -> AgentState:
    return {"session_id": "session-1", "user_id": 1, "messages": []}


def _runtime() -> Runtime[CueContext]:
    return Runtime(
        context=CueContext(
            session=None,  # type: ignore[arg-type]
            user_id=1,
            chat_session_id=uuid.uuid4(),
            address_id="addr-1",
        )
    )


def _order(
    status: OrderStatus = OrderStatus.OUT_FOR_DELIVERY,
    order_id: str = "ord-1",
    items: list[str] | None = None,
    total: Decimal | None = None,
) -> OrderListItem:
    return OrderListItem(
        order_id=order_id,
        status=status,
        placed_at="2026-07-28T10:00:00Z",
        items=items if items is not None else ["Paneer", "Butter"],
        total=total,
    )


def _stub_model(
    monkeypatch: pytest.MonkeyPatch, replies: list[Any]
) -> tuple[_FakeChatModel, list[ModelRole]]:
    model = _FakeChatModel(replies)
    roles: list[ModelRole] = []

    def _get_chat_model(role: ModelRole) -> _FakeChatModel:
        roles.append(role)
        return model

    monkeypatch.setattr(node_module, "get_chat_model", _get_chat_model)
    return model, roles


def _stub_orders(
    monkeypatch: pytest.MonkeyPatch, orders: list[OrderListItem]
) -> list[tuple[object, int]]:
    calls: list[tuple[object, int]] = []

    async def _list(session: object, user_id: int) -> list[OrderListItem]:
        calls.append((session, user_id))
        return orders

    monkeypatch.setattr(orders_service, "list_orders_throttled", _list)
    return calls


def _prompt_text(model: _FakeChatModel) -> str:
    return "\n".join(str(message.content) for message in model.prompts[0])


# --- no orders -------------------------------------------------------------


async def test_no_recent_orders_answers_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model handed an empty payload is exactly the one that invents an order.
    model, _ = _stub_model(monkeypatch, [])
    _stub_orders(monkeypatch, [])

    update = await order_status_node(_state(), _runtime())

    assert model.calls == 0
    assert str(update["messages"][0].content) == NO_ORDERS_MESSAGE


# --- which order ------------------------------------------------------------


async def test_an_undelivered_order_wins_over_a_newer_delivered_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "where is my order" is about the one that has not arrived, even when a
    # later order has already been delivered.
    model, _ = _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    _stub_orders(
        monkeypatch,
        [
            _order(OrderStatus.DELIVERED, order_id="newer"),
            _order(OrderStatus.PREPARING, order_id="still-coming"),
        ],
    )

    await order_status_node(_state(), _runtime())

    assert "being prepared" in _prompt_text(model)
    assert "delivered" not in _prompt_text(model)


async def test_with_no_active_order_the_newest_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _stub_model(monkeypatch, [AIMessage(content="It arrived.")])
    _stub_orders(
        monkeypatch,
        [
            _order(OrderStatus.DELIVERED, order_id="newest"),
            _order(OrderStatus.CANCELLED, order_id="older"),
        ],
    )

    await order_status_node(_state(), _runtime())

    assert "delivered" in _prompt_text(model)
    assert "cancelled" not in _prompt_text(model)


# --- the grounding facts ----------------------------------------------------


async def test_the_role_requested_is_order_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No model id or effort literal belongs in a node; the role is the contract.
    _, roles = _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    _stub_orders(monkeypatch, [_order()])

    await order_status_node(_state(), _runtime())

    assert roles == [ModelRole.ORDER_STATUS]


async def test_the_prompt_carries_the_mapped_status_never_a_raw_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    _stub_orders(monkeypatch, [_order(OrderStatus.OUT_FOR_DELIVERY)])

    await order_status_node(_state(), _runtime())

    prompt = _prompt_text(model)
    assert "out for delivery" in prompt
    # The closed enum's value never reaches the model as a raw token to reword.
    assert "OUT_FOR_DELIVERY" not in prompt


async def test_the_prompt_forbids_inventing_an_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # There is no ETA to give: track_order needs delivery-address coordinates,
    # and Swiggy withholds them from get_addresses. An invented delivery time is
    # the worst thing this node could produce.
    model, _ = _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    _stub_orders(monkeypatch, [_order()])

    await order_status_node(_state(), _runtime())

    prompt = _prompt_text(model).lower()
    assert "never state, estimate, or imply an arrival time" in prompt
    assert "live eta available: no" in prompt


async def test_a_long_item_list_is_truncated_for_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    _stub_orders(monkeypatch, [_order(items=[f"item-{index}" for index in range(9)])])

    await order_status_node(_state(), _runtime())

    prompt = _prompt_text(model)
    assert "item-4" in prompt
    assert "item-5" not in prompt
    assert "and 4 more" in prompt


async def test_the_read_is_scoped_to_the_runtime_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_model(monkeypatch, [AIMessage(content="On its way.")])
    calls = _stub_orders(monkeypatch, [_order()])

    await order_status_node(_state(), _runtime())

    assert [user_id for _session, user_id in calls] == [1]


# --- the reply --------------------------------------------------------------


async def test_the_models_sentence_is_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_model(monkeypatch, [AIMessage(content="Your paneer is out for delivery.")])
    _stub_orders(monkeypatch, [_order()])

    update = await order_status_node(_state(), _runtime())

    assert set(update.keys()) == {"messages"}
    assert str(update["messages"][0].content) == "Your paneer is out for delivery."


async def test_an_empty_completion_falls_back_to_the_deterministic_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The facts are already in hand, so an empty completion costs the phrasing
    # and nothing else - it must not leave the turn with no reply at all.
    _stub_model(monkeypatch, [AIMessage(content="   ")])
    _stub_orders(monkeypatch, [_order(OrderStatus.PREPARING)])

    update = await order_status_node(_state(), _runtime())

    reply = str(update["messages"][0].content)
    assert "being prepared" in reply
    assert "Orders tab" in reply


async def test_an_expired_link_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reconnecting Swiggy is the user's action; `stream_turn` names it in an
    # error event. Catching it here would replace that with a vague apology.
    _stub_model(monkeypatch, [])

    async def _raise(_session: object, _user_id: int) -> list[OrderListItem]:
        raise InstamartAuthError

    monkeypatch.setattr(orders_service, "list_orders_throttled", _raise)

    with pytest.raises(InstamartAuthError):
        await order_status_node(_state(), _runtime())
