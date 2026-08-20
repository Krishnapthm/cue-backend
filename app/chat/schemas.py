from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.schemas import GeneratedRecipe, MatchResult
from app.cart.schemas import MatchStatus
from app.instamart.schemas import ProductRating


class MessageKind(StrEnum):
    """The shape of a message body, mirrored from `ck_chat_message_kind_allowed`."""

    TEXT = "text"
    IMAGE = "image"
    CHECKLIST = "checklist"
    CART_READY = "cart_ready"
    RECIPE = "recipe"


class MessageRole(StrEnum):
    """Who authored a chat message, mirrored from `ck_chat_message_role_allowed`."""

    USER = "user"
    ASSISTANT = "assistant"


class ImageMessagePayload(BaseModel):
    """The `payload` of a `kind='image'` message (CUE-88).

    One shape, because there is one upload path. The design offers two attach
    tiles - camera and photo library - but both produce the same uploaded
    Supabase Storage object, so the backend has one contract, not two.

    `object_path` is the storage object path the client uploaded to, e.g.
    `recipes/u1/8f2c.jpg`. It is what reaches `AgentState.image_object_path`
    and, through it, `parse_recipe_photo_node`'s image fetch.
    """

    object_path: str = Field(min_length=1)


class RecipeStepPayload(BaseModel):
    """One step of a persisted recipe card, as the client renders it.

    A projection of `agent.schemas.RecipeStep` rather than the model itself,
    for the `index` alone: cooking mode navigates by position and saves the
    user's place as a number, so the position has to be a field on the wire and
    not the reader's array offset.

    `index` is **1-based and stable**. A saved cooking position is an index into
    this list, so nothing may renumber it - not a later reordering, not a
    filtering of steps on the way out.
    """

    index: int = Field(ge=1)
    title: str
    instructions: list[str]
    duration_seconds: int | None = None


class RecipeCardPayload(BaseModel):
    """The `payload` of a `kind='recipe'` message: the full recipe card.

    Persisted at the end of a cart turn so the reveal card and cooking mode
    both work on a cold start, hours after the order was placed, with no new
    endpoint and no checkpoint read. The recipe otherwise exists only inside
    the LangGraph checkpoint, which is an opaque blob keyed by thread.

    `ui` is a discriminator for the same reason `ChecklistInterrupt.ui` is one:
    the client routes on an explicit field, never on the payload's shape.

    `method_summary` rides along beside `steps` because it is what the
    collapsed card shows as its clipped preview - it is not the steps in prose
    form, and one does not substitute for the other.
    """

    ui: Literal["recipe"] = "recipe"
    dish_name: str
    estimated_time_minutes: int
    servings: int | None = None
    difficulty: str | None = None
    method_summary: str
    steps: list[RecipeStepPayload]

    @classmethod
    def from_recipe(cls, recipe: GeneratedRecipe) -> Self:
        """Project a generated recipe onto the card the transcript stores.

        Numbering happens here, once, so `index` is assigned in exactly one
        place rather than by each reader counting for itself.

        Args:
            recipe: The recipe the turn generated.

        Returns:
            The card payload, its steps numbered from 1 in cooking order.
        """
        return cls(
            dish_name=recipe.dish_name,
            estimated_time_minutes=recipe.estimated_time_minutes,
            servings=recipe.servings,
            difficulty=recipe.difficulty,
            method_summary=recipe.method_summary,
            steps=[
                RecipeStepPayload(
                    index=position,
                    title=step.title,
                    instructions=step.instructions,
                    duration_seconds=step.duration_seconds,
                )
                for position, step in enumerate(recipe.steps, start=1)
            ],
        )


class CreateMessageRequest(BaseModel):
    """A message to append to a chat session's transcript.

    `step_index` is the 1-based recipe step the user is looking at in cooking
    mode, sent alongside the question they typed. It is what makes the
    `cooking_question` turn path reachable (CUE-120) - a question asked mid-cook
    otherwise enters the recipe path and recomposes the user's cart underneath
    them. It is not persisted on the message: it describes where the user is,
    not what they said, and the transcript records the latter.
    """

    role: MessageRole
    kind: MessageKind = MessageKind.TEXT
    content: str | None = None
    payload: dict[str, Any] | None = None
    step_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_body(self) -> Self:
        """Mirror the `ck_chat_message_body` DB constraint client-side.

        `kind="text"` requires `content`; every other kind requires `payload`.
        Enforced here so a violation returns 422 before it ever reaches the
        database constraint.

        A `kind="image"` payload is validated further, against
        `ImageMessagePayload`: that turn runs the vision path, and an
        unusable payload should be a 422 at the boundary rather than a node
        raising several seconds into a turn the user is watching.
        """
        if self.kind is MessageKind.TEXT and self.content is None:
            raise ValueError("content is required when kind is 'text'.")
        if self.kind is not MessageKind.TEXT and self.payload is None:
            raise ValueError("payload is required when kind is not 'text'.")
        if self.kind is MessageKind.IMAGE and self.payload is not None:
            ImageMessagePayload.model_validate(self.payload)
        return self

    def image_object_path(self) -> str | None:
        """Return the uploaded object path, if this is an image message.

        The extraction the graph's boundary needs, kept on the request model so
        the payload is parsed through `ImageMessagePayload` in one place rather
        than indexed as a raw dict by every caller.
        """
        if self.kind is not MessageKind.IMAGE or self.payload is None:
            return None
        return ImageMessagePayload.model_validate(self.payload).object_path


class MessageResponse(BaseModel):
    """A persisted chat message, as returned in a transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    kind: MessageKind
    content: str | None
    payload: dict[str, Any] | None
    created_at: datetime


class MessageExchange(BaseModel):
    """One conversational turn: what the user sent and what the agent replied.

    `assistant_message` is `None` whenever the turn ran no agent - any kind
    other than `text`, or any role other than `user`. That is a normal
    outcome, not an error: those turns persist exactly as they always have.
    """

    user_message: MessageResponse
    assistant_message: MessageResponse | None


class SessionSummary(BaseModel):
    """A chat session as it appears in the Recents list (R8.1)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    selected_address_id: str | None
    updated_at: datetime


class UpdateSessionRequest(BaseModel):
    """A mutable chat-session setting supplied by the client."""

    selected_address_id: str = Field(min_length=1, max_length=100)


class SessionDetail(BaseModel):
    """A chat session with its full ordered message transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    selected_address_id: str | None
    messages: list[MessageResponse]


class StreamEventType(StrEnum):
    """The named SSE events a streamed turn can emit.

    This enum *is* the wire contract. LangGraph's own chunk shapes are an
    internal detail that changes between versions and must never reach a
    client, so every chunk is translated into one of these before it is sent.
    """

    TOKEN = "token"
    MATCH = "match"
    STAGE = "stage"
    INTERRUPT = "interrupt"
    ERROR = "error"
    DONE = "done"


class TokenEvent(BaseModel):
    """A fragment of assistant prose, to be appended as it arrives.

    Only nodes on `agent.graph.PROSE_NODES` produce these. Every other model
    call in the graph emits structured JSON, which is internal.
    """

    event: Literal[StreamEventType.TOKEN] = StreamEventType.TOKEN
    text: str


class MatchEvent(BaseModel):
    """One ingredient resolved, emitted the moment its worker finishes.

    `ingredient_name` is a stable key, not a position. The fan-out's workers
    finish out of order while the UI lists ingredients in recipe order, so the
    client fills the row with this name in place rather than appending. The
    parallelism is the point; serializing the workers to make the stream
    ordered would trade the feature for the convenience.
    """

    event: Literal[StreamEventType.MATCH] = StreamEventType.MATCH
    ingredient_name: str
    status: MatchStatus
    spin_id: str | None = None
    product_name: str | None = None
    pack_size: str | None = None
    unit_price: Decimal | None = None
    image_url: str | None = None
    rating: ProductRating | None = None
    quantity: int | None = None
    substitution_reason: str | None = None

    @classmethod
    def from_match(cls, match: MatchResult) -> Self:
        """Build the wire event for one resolved ingredient."""
        return cls(**match.model_dump())


class StageEvent(BaseModel):
    """The turn moved on to another node, so the UI can change its label."""

    event: Literal[StreamEventType.STAGE] = StreamEventType.STAGE
    node: str


class InterruptEvent(BaseModel):
    """The turn paused and is waiting on the user.

    `payload` is whatever the node passed to `interrupt()` - the checklist to
    confirm, the substitution to approve. `id` identifies which interrupt is
    being answered when more than one is pending.
    """

    event: Literal[StreamEventType.INTERRUPT] = StreamEventType.INTERRUPT
    id: str | None = None
    payload: Any = None


class StreamErrorCode(StrEnum):
    """Why a streamed turn stopped early, at the granularity the app acts on."""

    PROVIDER_AUTH = "provider_auth"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_REJECTED = "provider_rejected"
    AGENT_FAILED = "agent_failed"


class RecoveryAction(StrEnum):
    """What the app should offer the user after an `error` event."""

    RECONNECT_SWIGGY = "reconnect_swiggy"
    RETRY = "retry"


class ErrorEvent(BaseModel):
    """The turn failed after the response had already started.

    By the time this is emitted the status code is long gone, so a failure
    mid-stream cannot be a 500 - it has to be an event that names the action
    that recovers it, followed by a clean close.
    """

    event: Literal[StreamEventType.ERROR] = StreamEventType.ERROR
    code: StreamErrorCode
    message: str
    action: RecoveryAction


class DoneEvent(BaseModel):
    """The stream is over; nothing further will arrive on this connection.

    `interrupted` distinguishes a finished turn from a paused one: on a pause
    the client owes a decision, and can re-read it later from the session
    state endpoint if the connection drops before it is answered.
    """

    event: Literal[StreamEventType.DONE] = StreamEventType.DONE
    reply: str | None = None
    message_id: int | None = None
    interrupted: bool = False


ChatStreamEvent = Annotated[
    TokenEvent | MatchEvent | StageEvent | InterruptEvent | ErrorEvent | DoneEvent,
    Field(discriminator="event"),
]


class PendingInterrupt(BaseModel):
    """An interrupt the session is still waiting on."""

    id: str | None = None
    payload: Any = None


class SessionAgentState(BaseModel):
    """What the agent is waiting for on a session, if anything.

    SSE drops every time the app is backgrounded, and a user can close the app
    mid-checklist and come back the next day. The checkpointer has already
    persisted the pending interrupt, so this is a cheap read that covers
    reconnect and cold start with the same endpoint - without it, a dropped
    connection leaves the client with no way to discover a decision is owed.
    """

    pending_interrupt: PendingInterrupt | None = None
