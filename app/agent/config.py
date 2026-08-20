from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRole(StrEnum):
    """What a node needs a model *for*, rather than which model it wants.

    Nodes ask for a role; the id behind it lives in `AgentSettings` and is
    overridable by env var. That indirection is what keeps model choice
    swappable by config alone, and is why no node ever names a model.

    The graph's nodes have genuinely different cost/quality profiles, which is
    why one shared model id is no longer enough:

    * `ROUTER` - classification only, on every single turn. Cheap, and run at
      no reasoning effort; the router's job is a one-word label, not thought.
    * `RECIPE` - decides correctness. A wrong ingredient list becomes a wrong
      cart and then a wrong order, and the error is invisible until the user
      is at the stove, so this role buys the strongest model in the system.
    * `VISION` - reads a recipe photo into the same schema `RECIPE` produces,
      with the same correctness stakes.
    * `ORDER_STATUS` - turns an already-fetched tracking payload into one
      sentence. There is no reasoning to do, the output is short, and the user
      is waiting, so the cheapest fast model wins outright.
    * `TITLE` - condenses a resolved dish name into a short Recents label.
      It is best-effort metadata, never part of the user's turn, so it uses a
      cheap fast model with no reasoning effort.
    * `COOKING` - answers a question from someone standing at the stove
      ("can I use ghee instead?"). Unlike `ORDER_STATUS` this is not rewording
      a payload we already validated: the answer is cooking judgement, it is
      acted on immediately, and a wrong one ruins the dish the user has already
      bought the ingredients for. That is worth the strong model, one call per
      question.
    * `SMALL_TALK` - answers "thanks, that was delicious" in one short line.
      The cheapest role in the system and deliberately so: there is no
      judgement to make and nothing to get wrong except length, which the
      prompt asks for and the node enforces.

    Deterministic nodes (`normalize_ingredients`, `select_variant`,
    `propose_substitute`, `report_cart`, `refuse`) take no model at all and
    have no role here.
    """

    ROUTER = "router"
    RECIPE = "recipe"
    VISION = "vision"
    ORDER_STATUS = "order_status"
    TITLE = "title"
    COOKING = "cooking"
    SMALL_TALK = "small_talk"


class ReasoningEffort(StrEnum):
    """How hard a reasoning-capable model should think before answering.

    A closed enum rather than a free string: the value is passed straight to
    the provider, and a typo would be a runtime provider error on the first
    real turn rather than a config-load failure at startup.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass(frozen=True)
class ModelChoice:
    """The resolved model id (and effort, if any) behind one `ModelRole`.

    `reasoning_effort` is `None` for roles whose model does not offer the
    dial, so `providers.get_chat_model` can pass the kwarg only where it
    means something rather than sending an unsupported argument.
    """

    model_id: str
    reasoning_effort: ReasoningEffort | None = None


class AgentSettings(BaseSettings):
    """LangGraph agent runtime settings.

    Model provider choice is an open decision (PRD Section 12) and must stay
    swappable via config alone - never hard-coded in the graph or its nodes.
    Model *ids* follow the same rule, per `ModelRole`: every role's id, and
    the router's reasoning effort, are settings and therefore env vars
    (`AGENT_MODEL_ROUTER`, `AGENT_MODEL_RECIPE`, ...).
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Selects which langchain-core BaseChatModel `providers.get_chat_model`
    # returns. PRD Section 12 is settled: OpenAI. Anthropic stays wired up so
    # the seam is a live two-provider choice rather than a dead branch.
    MODEL_PROVIDER: Literal["openai", "anthropic"] = "openai"

    # Per-role model ids. Unlike the single MODEL_NAME these replace, these
    # carry defaults: the assignment below is a costed decision (see CUE-85),
    # not a deployment preference, so a deployment that overrides one should
    # be doing it deliberately rather than being forced to restate all three.
    #
    # Router: nano over 4o-mini. The headline input price is misleading -
    # every turn re-sends a fixed system prompt, so most input tokens are
    # *cached* ones, and nano caches at $0.02/M against 4o-mini's $0.075/M.
    # On a realistic router call that is ~40% cheaper per call despite nano
    # costing more per fresh token, and it adds the effort dial below.
    MODEL_ROUTER: str = "gpt-5.4-nano-2026-03-17"
    # Recipe and vision: luna, because these two decide correctness. The delta
    # over nano is ~$0.0045/turn - the cheapest correctness insurance in the
    # system - and luna's Feb 2026 cutoff is the most useful one for Indian
    # grocery vocabulary, brand names, and pack conventions.
    MODEL_RECIPE: str = "gpt-5.6-luna"
    MODEL_VISION: str = "gpt-5.6-luna"
    # Order status: nano, for the same cached-input reason as the router. This
    # node rewrites a structured payload the service layer already validated
    # into one sentence - the model is doing wording, not judgement.
    MODEL_ORDER_STATUS: str = "gpt-5.4-nano-2026-03-17"
    # Session titling is a low-stakes 2-4 word summarisation job. Use the
    # configured OpenAI provider's inexpensive, low-latency model; deployments
    # can replace it with AGENT_MODEL_TITLE when product needs change.
    MODEL_TITLE: str = "gpt-4o-mini"
    # Cooking answers: luna, for the correctness reason `ModelRole.COOKING`
    # gives. The user is waiting, but this node is on `PROSE_NODES` so its
    # tokens stream - perceived latency is the first token, not the last.
    MODEL_COOKING: str = "gpt-5.6-luna"
    # Small talk: nano, for the same cached-input reason as the router. One
    # short warm line back to "thanks, that was delicious" is the least
    # demanding generation in the system, and it is on `PROSE_NODES` so it
    # streams.
    MODEL_SMALL_TALK: str = "gpt-5.4-nano-2026-03-17"
    # The router emits a single label from an explicit rubric; reasoning
    # tokens buy nothing there and are billed at output rates.
    MODEL_ROUTER_REASONING_EFFORT: ReasoningEffort = ReasoningEffort.NONE
    # Same reasoning, and the stronger one: the user is waiting on this reply,
    # so latency spent thinking about a one-sentence status is latency wasted.
    MODEL_ORDER_STATUS_REASONING_EFFORT: ReasoningEffort = ReasoningEffort.NONE
    # And again: there is nothing to think about in "glad it turned out well",
    # and reasoning tokens are billed at output rates for a one-line reply.
    MODEL_SMALL_TALK_REASONING_EFFORT: ReasoningEffort = ReasoningEffort.NONE

    # Checkpointer connection pool (CUE-93). This pool is the graph's, and it
    # is **separate from SQLAlchemy's** (`DATABASE_POOL_SIZE`): the checkpointer
    # speaks psycopg, the app speaks asyncpg, and both count against the same
    # Postgres `max_connections`. Size them together, not independently.
    #
    # Why these numbers. A pooled checkpointer borrows a connection for the
    # duration of one checkpoint read or write - a few milliseconds - and
    # returns it, so the pool is sized by *checkpoint operations in flight*,
    # not by open SSE streams. That is the whole point of the change: before
    # it, every request opened its own connection and a long-lived stream
    # pinned it for the entire turn, so a handful of concurrent users
    # exhausted the database. A turn is a few super-steps and therefore a few
    # short writes, so 10 covers far more concurrent turns than 10.
    #
    # `min_size` keeps a couple warm so the first turn after an idle period
    # does not pay connection setup on the user's latency.
    CHECKPOINTER_POOL_MIN_SIZE: int = 2
    CHECKPOINTER_POOL_MAX_SIZE: int = 10

    # Supabase project base URL, used to build recipe-photo object URLs (see
    # `app.agent.storage.SupabaseImageStore`). Optional so the app still
    # imports without it configured; `SupabaseImageStore.load` raises a clear
    # error if it is unset when actually invoked.
    SUPABASE_URL: str | None = None
    # Supabase Storage bucket that recipe photo uploads land in.
    RECIPE_IMAGE_BUCKET: str = "recipe-images"

    def model_for(self, role: ModelRole) -> ModelChoice:
        """Resolve one role to the model id (and effort) configured for it.

        Args:
            role: The role a node is asking for.

        Returns:
            The configured `ModelChoice`. `reasoning_effort` is only set for
            roles whose model offers the dial - luna does not, so `RECIPE`
            and `VISION` return `None` for it.
        """
        match role:
            case ModelRole.ROUTER:
                return ModelChoice(
                    model_id=self.MODEL_ROUTER,
                    reasoning_effort=self.MODEL_ROUTER_REASONING_EFFORT,
                )
            case ModelRole.RECIPE:
                return ModelChoice(model_id=self.MODEL_RECIPE)
            case ModelRole.VISION:
                return ModelChoice(model_id=self.MODEL_VISION)
            case ModelRole.ORDER_STATUS:
                return ModelChoice(
                    model_id=self.MODEL_ORDER_STATUS,
                    reasoning_effort=self.MODEL_ORDER_STATUS_REASONING_EFFORT,
                )
            case ModelRole.TITLE:
                # gpt-4o-mini does not support the reasoning_effort request
                # argument. Passing even "none" makes OpenAI reject title
                # generation with HTTP 400.
                return ModelChoice(model_id=self.MODEL_TITLE)
            case ModelRole.COOKING:
                # luna does not offer the reasoning-effort dial, same as
                # RECIPE and VISION.
                return ModelChoice(model_id=self.MODEL_COOKING)
            case ModelRole.SMALL_TALK:
                return ModelChoice(
                    model_id=self.MODEL_SMALL_TALK,
                    reasoning_effort=self.MODEL_SMALL_TALK_REASONING_EFFORT,
                )


@lru_cache
def get_agent_settings() -> AgentSettings:
    """Return the cached agent settings."""
    return AgentSettings()


agent_settings = get_agent_settings()
