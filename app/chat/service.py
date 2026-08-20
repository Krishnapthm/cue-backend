"""Chat session and message persistence (R8.1/R8.3, CUE-20).

This is the display transcript only - the LangGraph checkpointer (CUE-21)
owns the agent's own state under the same `chat_session.id` as its
`thread_id`. The two are deliberately decoupled; see `app/models/chat.py`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import CueContext
from app.agent.exceptions import RecipeGenerationError
from app.agent.graph import PROSE_NODES, CueGraph, thread_config
from app.agent.schemas import (
    CartReport,
    ChecklistDecision,
    GeneratedRecipe,
    MatchResult,
    ScratchChoiceDecision,
)
from app.agent.state import AgentState
from app.chat.constants import ADDRESS_REQUIRED_MESSAGE
from app.chat.exceptions import (
    ChatSessionNotFoundError,
    InvalidChecklistAnswerError,
    NoPendingChecklistError,
)
from app.chat.schemas import (
    ChatStreamEvent,
    CreateMessageRequest,
    DoneEvent,
    ErrorEvent,
    InterruptEvent,
    MatchEvent,
    MessageKind,
    MessageRole,
    PendingInterrupt,
    RecipeCardPayload,
    RecoveryAction,
    SessionAgentState,
    StageEvent,
    StreamErrorCode,
    TokenEvent,
)
from app.exceptions import AppError
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.models.chat import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

#: The key LangGraph reports a pause under in `updates` mode. Spelled out
#: rather than imported: `langgraph.constants.INTERRUPT` is private as of
#: LangGraph v1 and slated for removal, and this is a stable wire-level name.
INTERRUPT_KEY = "__interrupt__"


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the checklist an `ainvoke` result paused on, if it paused.

    Args:
        result: What `graph.ainvoke` returned for the turn.

    Returns:
        The first interrupt's payload, or `None` when the turn ran to the end.
    """
    interrupts = result.get(INTERRUPT_KEY)
    if not interrupts:
        return None
    payload = getattr(interrupts[0], "value", None)
    return payload if isinstance(payload, dict) else None


def _recipe(value: Any) -> GeneratedRecipe | None:
    """Read a `recipe` state value, whatever shape it comes back in.

    Accepted from both shapes for the reason `_cart_report` is: state replayed
    through the checkpointer arrives as plain JSON, state straight off an
    `ainvoke` is still the model instance.

    A recipe that no longer validates is dropped rather than raised on. This
    runs *after* the cart has been pushed to Swiggy and the `cart_ready`
    message written, so failing the turn here would report a completed order as
    a failure over a card the user has not seen yet. A session whose recipe
    predates a schema change is exactly this case.
    """
    if value is None:
        return None
    try:
        return GeneratedRecipe.model_validate(value)
    except ValidationError:
        logger.warning("Ignoring an unreadable recipe on the turn's state.")
        return None


def _cart_report(value: Any) -> CartReport | None:
    """Read a `cart_report` state value, whatever shape it comes back in.

    Validated rather than cast: state that has been through the checkpointer
    comes back as plain JSON, while state straight off an `ainvoke` is still
    the model instance. Both are accepted; anything else is not a report.
    """
    if value is None:
        return None
    try:
        return CartReport.model_validate(value)
    except ValidationError:
        logger.warning("Ignoring an unreadable cart_report on the turn's state.")
        return None


async def create_session(session: AsyncSession, user_id: int) -> ChatSession:
    """Create a new, untitled chat session for `user_id`.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The newly created session.
    """
    chat_session = ChatSession(user_id=user_id)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


async def set_selected_address(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    selected_address_id: str,
) -> ChatSession:
    """Persist the delivery address selected for a user's chat session.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The chat session to update.
        selected_address_id: The external Swiggy address ID to use for carts.

    Returns:
        The updated chat session.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
    """
    chat_session = await get_session(session, user_id, session_id)
    chat_session.selected_address_id = selected_address_id
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


async def list_sessions(session: AsyncSession, user_id: int) -> list[ChatSession]:
    """Return `user_id`'s chat sessions for the Recents list (R8.1).

    Most-recently-updated first, via `ix_chat_session_user_recents`. No date
    grouping or search - that is out of scope for CUE-20.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The user's sessions, ordered by `updated_at` descending.
    """
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_session(
    session: AsyncSession, user_id: int, session_id: uuid.UUID
) -> ChatSession:
    """Return a chat session, scoped to its owner.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to fetch.

    Returns:
        The matching session.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist, or exists
            but belongs to a different user - both cases return 404, never
            leaking another user's data.
    """
    stmt = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    )
    result = await session.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise ChatSessionNotFoundError
    return chat_session


async def list_messages(
    session: AsyncSession, session_id: uuid.UUID
) -> list[ChatMessage]:
    """Return a session's transcript in display order.

    Ordered by `id`, not `created_at`, per `ix_chat_message_session` - `id`
    is monotonic and collision-free, `created_at` is not guaranteed to be.
    Does not check ownership; callers must authorize `session_id` first via
    `get_session`.

    Args:
        session: An active database session.
        session_id: The session whose messages to fetch.

    Returns:
        The session's messages, oldest first.
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def append_message(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> ChatMessage:
    """Append a message to a session's transcript and resurface it in Recents.

    Bumps `chat_session.updated_at` so the session sorts to the top of
    Recents (R8.1) via the shared `trg_chat_session_updated_at` trigger,
    which stamps `now()` on any UPDATE regardless of the value written here.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to.
        request: The message to append; already validated against the
            `ck_chat_message_body` shape by `CreateMessageRequest`.

    Returns:
        The newly persisted message.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
    """
    await get_session(session, user_id, session_id)

    message = ChatMessage(
        session_id=session_id,
        role=request.role.value,
        kind=request.kind.value,
        content=request.content,
        payload=request.payload,
    )
    session.add(message)

    bump_stmt = (
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=func.now())
    )
    await session.execute(bump_stmt)

    await session.commit()
    await session.refresh(message)
    return message


#: The message kinds a user can start a turn with. Text is a dish name or a
#: question; an image is an uploaded recipe photo, which CUE-88 routes to the
#: vision path. Everything else - a `checklist` or `cart_ready` append, or
#: anything authored by the assistant - is a transcript write, and must never
#: burn a model call.
_AGENT_KINDS = frozenset({MessageKind.TEXT, MessageKind.IMAGE})


def _runs_the_agent(request: CreateMessageRequest) -> bool:
    """Return whether this message should be handed to the agent."""
    return request.role is MessageRole.USER and request.kind in _AGENT_KINDS


def _reply_text(messages: list[BaseMessage]) -> str | None:
    """Return the content of the last `AIMessage` in a graph result."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content)
            return content or None
    return None


def _turn_state(
    user_id: int, session_id: uuid.UUID, request: CreateMessageRequest
) -> AgentState:
    """Build the state literal one turn starts from.

    Only the user's new turn goes in: the rest of the transcript is replayed
    by the checkpointer under the same `thread_id`.

    An image turn carries its uploaded object path onto
    `image_object_path` - this is the `ChatMessage.payload` -> state extraction
    at the app/graph boundary that CUE-23 deferred. `route_turn` branches on
    that field alone, which is exactly why it is set here from the upload rather
    than from anything the user can type.

    It is written on every turn, `None` included: the field survives in the
    checkpoint, so leaving a previous turn's path in place would route the next
    text turn back into the vision path and re-read a stale photo.

    `active_step_index` is carried the same way and for the same reason
    (CUE-120). It is what makes a turn eligible for the cooking path, and
    leaving a previous turn's value in place would keep offering that path
    after the user has closed cooking mode - so a turn sent without one clears
    it rather than inheriting it.
    """
    return {
        "session_id": str(session_id),
        "user_id": user_id,
        "messages": [HumanMessage(content=request.content or "")],
        "image_object_path": request.image_object_path(),
        "active_step_index": request.step_index,
    }


async def _turn_context(
    session: AsyncSession, user_id: int, session_id: uuid.UUID
) -> CueContext | None:
    """Build the runtime context for one turn, if the session can run one.

    The context is built fresh per invocation and never checkpointed: it
    carries this request's `AsyncSession`, so a node's writes land in the same
    unit of work as the request that triggered them. See
    `app/agent/context.py`.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session being run.

    Returns:
        The context, or `None` when the session has no delivery address
        selected yet - a precondition of running a turn at all, since Swiggy
        binds a cart to an address.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
    """
    chat_session = await get_session(session, user_id, session_id)
    if not chat_session.selected_address_id:
        return None
    return CueContext(
        session=session,
        user_id=user_id,
        chat_session_id=session_id,
        address_id=chat_session.selected_address_id,
    )


async def _persist_reply(
    session: AsyncSession, user_id: int, session_id: uuid.UUID, reply: str
) -> ChatMessage:
    """Append one assistant text message to a session's transcript."""
    return await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT, kind=MessageKind.TEXT, content=reply
        ),
    )


async def _persist_checklist(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    payload: dict[str, Any],
) -> ChatMessage:
    """Append the paused turn's checklist to the transcript.

    `MessageKind.CHECKLIST` and `CreateMessageRequest`'s payload requirement for
    it both already existed - the schema anticipated this - so a pause needs no
    new message shape, only this write.
    """
    return await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT, kind=MessageKind.CHECKLIST, payload=payload
        ),
    )


async def _persist_cart_report(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    report: CartReport,
) -> ChatMessage:
    """Append the finished cart to the transcript as a `CART_READY` message.

    Written here rather than in `report_cart` for the same reason the checklist
    is: every transcript write belongs to one owner. `content` carries the
    node's summary line so a client that renders nothing but text still says
    something useful; `payload` carries the card.
    """
    return await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT,
            kind=MessageKind.CART_READY,
            content=report.summary,
            payload=report.model_dump(mode="json"),
        ),
    )


async def _persist_recipe_card(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    recipe: GeneratedRecipe,
) -> ChatMessage:
    """Append the turn's recipe to the transcript as a `RECIPE` message.

    Written here rather than in a graph node for the same reason the checklist
    and the cart card are: every transcript write belongs to one owner, and a
    node reaching into `chat.service` would close an import cycle through
    `agent.graph`.

    It is the last message of the turn, immediately after `cart_ready`, which
    is the product framing - "once the cart is ready, show the full recipe".
    `content` carries the dish name so a client that renders nothing but text
    still says something useful; `payload` carries the card.
    """
    return await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT,
            kind=MessageKind.RECIPE,
            content=recipe.dish_name,
            payload=RecipeCardPayload.from_recipe(recipe).model_dump(mode="json"),
        ),
    )


def _interrupt_answer(request: CreateMessageRequest) -> dict[str, Any] | None:
    """Read an inbound message as a structured interrupt answer, if it is one.

    `kind='checklist'` carries both the existing checklist answer and the
    scratch-choice answer. The pending interrupt's discriminated `ui` field
    chooses the exact schema later in `_resume_input`; validating it here
    would guess which card the session is actually waiting on.

    Args:
        request: The inbound message.

    Returns:
        The raw answer payload, or `None` when this message is not an interrupt
        answer.
    """
    if (
        request.role is not MessageRole.USER
        or request.kind is not MessageKind.CHECKLIST
    ):
        return None
    if request.payload is None or not ({"have", "choice"} & request.payload.keys()):
        raise InvalidChecklistAnswerError
    return request.payload


async def _resume_input(
    graph: CueGraph, session_id: uuid.UUID, answer: dict[str, Any]
) -> Command[Any]:
    """Build the resume input for a session that is actually paused.

    The pending interrupt is read from the checkpointer rather than tracked in a
    column of our own: the checkpointer is already the source of truth, and a
    second one would drift. It is the same read the `/state` endpoint does.

    Args:
        graph: The compiled agent graph for this request.
        session_id: The session being resumed; its `str()` is the `thread_id`.
        answer: The user's structured card decision.

    Returns:
        `Command(resume=...)` - never a plain state dict, which would not error
        but would start a fresh run and leave the session looking stuck.

    Raises:
        NoPendingChecklistError: Nothing is paused on this thread, so there is
            no interrupt for this answer to resume.
    """
    state = await pending_interrupt(graph, session_id)
    if state.pending_interrupt is None:
        raise NoPendingChecklistError
    payload = state.pending_interrupt.payload
    if not isinstance(payload, dict):
        raise InvalidChecklistAnswerError
    try:
        if payload.get("ui") == "checklist":
            return Command(resume=ChecklistDecision.model_validate(answer).model_dump())
        elif payload.get("ui") == "scratch_choice":
            return Command(
                resume=ScratchChoiceDecision.model_validate(answer).model_dump(
                    mode="json"
                )
            )
        else:
            raise ValueError("Unknown interrupt payload.")
    except (ValidationError, ValueError) as exc:
        raise InvalidChecklistAnswerError from exc


async def run_turn(
    session: AsyncSession,
    graph: CueGraph,
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> tuple[ChatMessage, ChatMessage | None]:
    """Persist an inbound message, run the agent on it, and persist the reply.

    The user's message is persisted first, and stays persisted even if the
    agent then fails: the user did send it, and rolling it back would lose
    input they typed. Ownership is checked before any of that (inside
    `append_message`), so an unauthorized request 404s without ever
    reaching the agent - and never burns a model call.

    The agent runs against `thread_id = str(session_id)`, so two sessions
    belonging to the same user keep entirely separate agent memory. That
    thread is the LangGraph checkpointer's own state, distinct from this
    display transcript; see the module docstring.

    A turn on a session with no delivery address selected is answered here,
    without invoking the graph: Swiggy binds a cart to an address, so the turn
    could not finish, and spending a model call on it would be waste. Address
    selection is a precondition, never something the agent decides.

    The whole turn is one blocking request. `stream_turn` is the streaming
    equivalent and is what the app uses; this stays for clients that want one
    payload, and for turns short enough not to need progress.

    Args:
        session: An active database session.
        graph: The compiled agent graph for this request.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to and run the agent against.
        request: The inbound message.

    Returns:
        The persisted user message, and the persisted assistant reply - or
        `None` when the turn ran no agent, or the agent produced no reply.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
        RecipeGenerationError: If the agent could not produce a recipe. The
            user's message stays persisted; the global handler maps this to
            502.
    """
    answer = _interrupt_answer(request)
    user_message = await append_message(session, user_id, session_id, request)
    if answer is None and not _runs_the_agent(request):
        return user_message, None

    context = await _turn_context(session, user_id, session_id)
    if context is None:
        prompt = await _persist_reply(
            session, user_id, session_id, ADDRESS_REQUIRED_MESSAGE
        )
        return user_message, prompt

    agent_input: Command[Any] | AgentState = (
        await _resume_input(graph, session_id, answer)
        if answer is not None
        else _turn_state(user_id, session_id, request)
    )
    result = await graph.ainvoke(
        agent_input,
        # The same `thread_config` on both calls: pause and resume must share a
        # `thread_id` or the resume silently becomes a new conversation.
        thread_config(str(session_id)),
        context=context,
    )

    interrupt_payload = _interrupt_payload(result)
    if interrupt_payload is not None:
        # A paused turn owes the user a decision, not a reply.
        checklist = await _persist_checklist(
            session, user_id, session_id, interrupt_payload
        )
        return user_message, checklist

    report = _cart_report(result.get("cart_report"))
    if report is not None:
        # A cart turn ends on its card, not on prose. This is also the only
        # thing a resumed turn produces, which is why it is checked first.
        cart_message = await _persist_cart_report(session, user_id, session_id, report)
        recipe = _recipe(result.get("recipe"))
        if recipe is not None:
            # The recipe card is appended *after* the cart card, and it is not
            # the turn's reply: the cart is still what the turn answered with,
            # and the recipe is the durable copy cooking mode reads back on a
            # cold start.
            await _persist_recipe_card(session, user_id, session_id, recipe)
        return user_message, cart_message

    if answer is not None:
        # A resume that produced no cart has nothing to say: everything in
        # `result["messages"]` is replayed history, so persisting the "last"
        # reply here would duplicate the recipe bubble from the turn that paused.
        return user_message, None

    reply = _reply_text(result["messages"])
    if reply is None:
        # Every branch of the graph emits a reply, so this means the graph
        # changed shape without this call site being updated. Persisting
        # nothing is better than persisting an empty assistant bubble.
        logger.error(
            "Agent produced no reply for session %s; persisting the user message only.",
            session_id,
        )
        return user_message, None

    assistant_message = await _persist_reply(session, user_id, session_id, reply)
    return user_message, assistant_message


#: Which domain failures become which `error` event. Anything not listed is a
#: bug rather than a condition the user can act on, and is left to raise.
_ERROR_EVENTS: dict[type[AppError], tuple[StreamErrorCode, RecoveryAction]] = {
    InstamartAuthError: (
        StreamErrorCode.PROVIDER_AUTH,
        RecoveryAction.RECONNECT_SWIGGY,
    ),
    InstamartTransportError: (
        StreamErrorCode.PROVIDER_UNAVAILABLE,
        RecoveryAction.RETRY,
    ),
    InstamartDomainError: (StreamErrorCode.PROVIDER_REJECTED, RecoveryAction.RETRY),
    RecipeGenerationError: (StreamErrorCode.AGENT_FAILED, RecoveryAction.RETRY),
}


def _error_event(exc: AppError) -> ErrorEvent | None:
    """Translate a domain error into the event that names its recovery."""
    for error_type, (code, action) in _ERROR_EVENTS.items():
        if isinstance(exc, error_type):
            return ErrorEvent(code=code, message=exc.detail, action=action)
    return None


def _token_event(chunk: Any) -> TokenEvent | None:
    """Build a token event from a `messages`-mode chunk, if it carries prose.

    LangGraph yields `(message_chunk, metadata)` here for *every* model call in
    the graph, including the structured-output ones whose tokens are JSON
    fragments of internal schemas. Only nodes on the `PROSE_NODES` allowlist
    are forwarded; see the note there on why it is an allowlist.
    """
    message, metadata = chunk
    if metadata.get("langgraph_node") not in PROSE_NODES:
        return None
    text = str(message.content)
    return TokenEvent(text=text) if text else None


def _custom_event(chunk: Any) -> MatchEvent | None:
    """Build a match event from a node's `get_stream_writer()` payload.

    Nodes write `MatchResult`-shaped payloads. Anything else is dropped rather
    than forwarded: the client's contract is this module's event types, not
    whatever a node happened to write.
    """
    try:
        return MatchEvent.from_match(MatchResult.model_validate(chunk))
    except ValidationError:
        logger.warning("Dropping unrecognized custom stream payload from a node.")
        return None


def _updates_events(chunk: dict[str, Any]) -> Iterator[StageEvent | InterruptEvent]:
    """Turn one `updates`-mode chunk into stage and interrupt events."""
    for node, node_update in chunk.items():
        if node == INTERRUPT_KEY:
            for interrupt in node_update:
                yield InterruptEvent(
                    id=getattr(interrupt, "id", None),
                    payload=getattr(interrupt, "value", None),
                )
            continue
        yield StageEvent(node=node)


def _messages_in(update: Any) -> list[BaseMessage]:
    """Return the messages a node's partial state update appended, if any."""
    if not isinstance(update, dict):
        return []
    messages = update.get("messages")
    return messages if isinstance(messages, list) else []


async def _checkpointed_recipe(
    graph: CueGraph, session_id: uuid.UUID
) -> GeneratedRecipe | None:
    """Read the session's recipe off the checkpoint.

    The streaming path needs this and the blocking one does not, because
    `astream` in `updates` mode yields *deltas*: the recipe was written by
    `generate_recipe` on the turn that paused at the checklist, and the turn
    that actually produces the cart is the resume, which re-runs no node that
    touches it. `ainvoke` returns the whole final state, so `run_turn` already
    has it.

    One read, once per cart turn, on the same handle `pending_interrupt` uses.
    It deliberately does not raise: a checkpoint that cannot be read costs the
    user a recipe card on an order that was still placed successfully.

    Args:
        graph: The compiled agent graph for this request.
        session_id: The session being run; its `str()` is the `thread_id`.

    Returns:
        The recipe on the session's state, or `None` when there is none.
    """
    snapshot = await graph.aget_state(thread_config(str(session_id)))
    return _recipe(snapshot.values.get("recipe") if snapshot.values else None)


async def stream_turn(
    session: AsyncSession,
    graph: CueGraph,
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> AsyncIterator[ChatStreamEvent]:
    """Run one turn, emitting typed events as the work happens.

    The blocking `run_turn` returns a single payload after several seconds of
    Swiggy calls and shows nothing in between. The chat design resolves
    ingredient rows one at a time as they come back, and the ingredient
    fan-out runs N parallel searches, so the designed screen cannot be built
    without this.

    Three LangGraph stream modes drive it: `messages` for prose tokens,
    `custom` for the per-ingredient payloads nodes write, and `updates` for
    stage changes and the `__interrupt__` pause. Every chunk is translated
    into one of this module's event types - raw LangGraph shapes are an
    internal contract that changes between versions and never reach a client.

    A domain failure after the first byte cannot be a status code: the
    response has already started. Those are emitted as an `error` event
    naming the action that recovers them, and the stream then closes cleanly.
    Anything not in `_ERROR_EVENTS` is a bug, and is left to raise.

    Args:
        session: An active database session.
        graph: The compiled agent graph for this request.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to and run the agent against.
        request: The inbound message.

    Yields:
        The turn's events, always ending with exactly one `done`.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`. Raised before anything is yielded, so the
            caller can still answer with a 404.
    """
    answer = _interrupt_answer(request)
    await append_message(session, user_id, session_id, request)
    if answer is None and not _runs_the_agent(request):
        yield DoneEvent()
        return

    context = await _turn_context(session, user_id, session_id)
    if context is None:
        prompt = await _persist_reply(
            session, user_id, session_id, ADDRESS_REQUIRED_MESSAGE
        )
        yield DoneEvent(reply=ADDRESS_REQUIRED_MESSAGE, message_id=prompt.id)
        return

    agent_input: Command[Any] | AgentState = (
        await _resume_input(graph, session_id, answer)
        if answer is not None
        else _turn_state(user_id, session_id, request)
    )

    replies: list[BaseMessage] = []
    interrupt_payload: dict[str, Any] | None = None
    report: CartReport | None = None
    recipe: GeneratedRecipe | None = None

    try:
        async for stream_mode, chunk in graph.astream(
            agent_input,
            thread_config(str(session_id)),
            context=context,
            stream_mode=["messages", "custom", "updates"],
        ):
            match stream_mode:
                case "messages":
                    if (token := _token_event(chunk)) is not None:
                        yield token
                case "custom":
                    if (match := _custom_event(chunk)) is not None:
                        yield match
                case "updates":
                    updates = cast("dict[str, Any]", chunk)
                    for node_update in updates.values():
                        replies.extend(_messages_in(node_update))
                        if isinstance(node_update, dict):
                            report = (
                                _cart_report(node_update.get("cart_report")) or report
                            )
                            recipe = _recipe(node_update.get("recipe")) or recipe
                    for update_event in _updates_events(updates):
                        if isinstance(update_event, InterruptEvent) and isinstance(
                            update_event.payload, dict
                        ):
                            interrupt_payload = update_event.payload
                        yield update_event
    except AppError as exc:
        error_event = _error_event(exc)
        if error_event is None:
            raise
        logger.warning(
            "Turn for session %s failed mid-stream: %s", session_id, exc.detail
        )
        yield error_event
        yield DoneEvent()
        return

    if interrupt_payload is not None:
        # A paused turn owes the user a decision, not a reply. The checklist is
        # persisted so the transcript still renders it after a reconnect or a
        # cold start - the SSE event alone would be lost with the connection.
        await _persist_checklist(session, user_id, session_id, interrupt_payload)
        yield DoneEvent(interrupted=True)
        return

    if report is not None:
        # A cart turn ends on its card. Persisted for the same reason the
        # checklist is: the SSE event alone dies with the connection, and the
        # transcript has to still render the cart on a cold start.
        cart_message = await _persist_cart_report(session, user_id, session_id, report)
        recipe = recipe or await _checkpointed_recipe(graph, session_id)
        if recipe is not None:
            await _persist_recipe_card(session, user_id, session_id, recipe)
        yield DoneEvent(reply=report.summary, message_id=cart_message.id)
        return

    reply = _reply_text(replies)
    if reply is None:
        logger.error(
            "Agent produced no reply for session %s; persisting the user message only.",
            session_id,
        )
        yield DoneEvent()
        return

    assistant_message = await _persist_reply(session, user_id, session_id, reply)
    yield DoneEvent(reply=reply, message_id=assistant_message.id)


async def pending_interrupt(
    graph: CueGraph, session_id: uuid.UUID
) -> SessionAgentState:
    """Return the interrupt a session is waiting on, if any.

    Read straight off the checkpointer, which has already persisted it, so
    this answers both a dropped SSE connection and a cold start the next day.

    Args:
        graph: The compiled agent graph for this request.
        session_id: The session to inspect; its `str()` is the `thread_id`.

    Returns:
        The pending interrupt, or an empty state when the session is idle.
    """
    snapshot = await graph.aget_state(thread_config(str(session_id)))
    interrupts = snapshot.interrupts
    if not interrupts:
        return SessionAgentState()
    first = interrupts[0]
    return SessionAgentState(
        pending_interrupt=PendingInterrupt(
            id=getattr(first, "id", None), payload=getattr(first, "value", None)
        )
    )
