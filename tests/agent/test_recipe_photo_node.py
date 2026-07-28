"""`parse_recipe_photo_node`: image-path extraction, structured-output retry,
and the shared `GeneratedRecipe` contract with `generate_recipe_node`.

These are unit tests - no real model and no real network call. `get_chat_model`
is monkeypatched to a fake `BaseChatModel`-shaped object (mirroring
`test_recipe_node.py`), and `get_image_store` is monkeypatched to a fake
`ImageStore` whose `.load` returns fixed bytes, so the suite runs fully
offline with no provider API key and no Supabase configuration.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.exceptions import OutputParserException

from app.agent.config import AgentSettings, ModelRole
from app.agent.exceptions import RecipeGenerationError
from app.agent.nodes import recipe as recipe_node
from app.agent.schemas import GeneratedRecipe, RecipeIngredient
from app.agent.state import AgentState
from app.agent.storage import SupabaseImageStore

_FAKE_IMAGE_BYTES = b"\xff\xd8\xff\xe0jpeg-ish-fixture-bytes"


class _FakeStructuredRunnable:
    """Stands in for `chat_model.with_structured_output(GeneratedRecipe)`.

    Pops one queued result per `ainvoke` call, in order, so a test can queue
    e.g. `[OutputParserException(...), a_valid_recipe]` to exercise the
    retry-then-succeed path.
    """

    def __init__(self, results: list[GeneratedRecipe | Exception]) -> None:
        self._results = list(results)

    async def ainvoke(self, _prompt: list[Any]) -> GeneratedRecipe:
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeChatModel:
    """Stands in for the `BaseChatModel` returned by `get_chat_model`."""

    def __init__(self, results: list[GeneratedRecipe | Exception]) -> None:
        self._results = results

    def with_structured_output(self, _schema: type) -> _FakeStructuredRunnable:
        return _FakeStructuredRunnable(self._results)


class _FakeImageStore:
    """Stands in for the `ImageStore` returned by `get_image_store`.

    Records every `object_path` passed to `.load` so tests can assert the
    node forwards the path from state unchanged.
    """

    def __init__(self, image_bytes: bytes = _FAKE_IMAGE_BYTES) -> None:
        self._image_bytes = image_bytes
        self.loaded_paths: list[str] = []

    async def load(self, object_path: str) -> bytes:
        self.loaded_paths.append(object_path)
        return self._image_bytes


def _state(image_object_path: str | None = "recipes/user-1/photo.jpg") -> AgentState:
    return {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [],
        "image_object_path": image_object_path,
    }


def _recipe(dish_name: str = "grandma's lasagna") -> GeneratedRecipe:
    return GeneratedRecipe(
        dish_name=dish_name,
        estimated_time_minutes=90,
        ingredients=[
            RecipeIngredient(name="lasagna sheets", quantity=250, unit="g"),
            RecipeIngredient(name="ground beef", quantity=500, unit="g"),
            RecipeIngredient(name="tomato sauce", quantity=400, unit="ml"),
        ],
        method_summary="Layer pasta, meat sauce, and cheese; bake until golden.",
    )


def _stub_chat_model(
    monkeypatch: pytest.MonkeyPatch, results: list[GeneratedRecipe | Exception]
) -> list[ModelRole]:
    """Stub the model seam, returning the roles the node asked it for."""
    roles: list[ModelRole] = []

    def _get_chat_model(role: ModelRole) -> _FakeChatModel:
        roles.append(role)
        return _FakeChatModel(results)

    monkeypatch.setattr(recipe_node, "get_chat_model", _get_chat_model)
    return roles


def _stub_image_store(
    monkeypatch: pytest.MonkeyPatch, store: _FakeImageStore | None = None
) -> _FakeImageStore:
    fake_store = store or _FakeImageStore()
    monkeypatch.setattr(recipe_node, "get_image_store", lambda: fake_store)
    return fake_store


async def test_parse_recipe_photo_node_asks_for_the_vision_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reading a photo carries the same correctness stakes as generating a
    # recipe from a dish name, and gets the same class of model - by role.
    roles = _stub_chat_model(monkeypatch, [_recipe()])
    _stub_image_store(monkeypatch)

    await recipe_node.parse_recipe_photo_node(_state())

    assert roles == [ModelRole.VISION]


async def test_parse_recipe_photo_node_returns_recipe_on_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    _stub_chat_model(monkeypatch, [recipe])
    fake_store = _stub_image_store(monkeypatch)

    update = await recipe_node.parse_recipe_photo_node(
        _state("recipes/user-1/photo.jpg")
    )

    # The node returns a partial state update, not a full AgentState.
    # Recipe-turn state persists in the checkpoint, so a new photo clears both
    # stale checklist marks and any prior ready-made choice/component.
    assert update == {
        "recipe": recipe,
        "have_marks": set(),
        "scratch_component": None,
        "scratch_choice": None,
    }
    assert update["recipe"].estimated_time_minutes == 90
    assert [i.name for i in update["recipe"].ingredients] == [
        "lasagna sheets",
        "ground beef",
        "tomato sauce",
    ]
    # The image store was called with the exact object path from state.
    assert fake_store.loaded_paths == ["recipes/user-1/photo.jpg"]


async def test_parse_recipe_photo_node_non_recipe_image_flows_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model itself is responsible for the empty-ingredients / best-effort
    # behaviour on a non-recipe photo (prompt design); here we just confirm
    # the node doesn't special-case or reject it - whatever GeneratedRecipe
    # the model produces flows straight onto state.
    non_recipe_result = GeneratedRecipe(
        dish_name="unrecognized photo",
        estimated_time_minutes=0,
        ingredients=[],
        method_summary="No recipe was recognized in this image.",
    )
    _stub_chat_model(monkeypatch, [non_recipe_result])
    _stub_image_store(monkeypatch)

    update = await recipe_node.parse_recipe_photo_node(_state())

    assert update["recipe"].ingredients == []
    assert "No recipe was recognized" in update["recipe"].method_summary


async def test_parse_recipe_photo_node_retries_once_on_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    _stub_chat_model(
        monkeypatch, [OutputParserException("could not parse model output"), recipe]
    )
    _stub_image_store(monkeypatch)

    update = await recipe_node.parse_recipe_photo_node(_state())

    # Recipe-turn state persists in the checkpoint, so a new photo clears both
    # stale checklist marks and any prior ready-made choice/component.
    assert update == {
        "recipe": recipe,
        "have_marks": set(),
        "scratch_component": None,
        "scratch_choice": None,
    }


async def test_parse_recipe_photo_node_raises_domain_error_after_two_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chat_model(
        monkeypatch,
        [
            OutputParserException("could not parse model output"),
            OutputParserException("still could not parse model output"),
        ],
    )
    _stub_image_store(monkeypatch)

    with pytest.raises(RecipeGenerationError):
        await recipe_node.parse_recipe_photo_node(_state())


async def test_parse_recipe_photo_node_raises_on_missing_image_object_path() -> None:
    with pytest.raises(ValueError, match="image_object_path"):
        await recipe_node.parse_recipe_photo_node(_state(None))


async def test_parse_recipe_photo_node_raises_on_empty_image_object_path() -> None:
    with pytest.raises(ValueError, match="image_object_path"):
        await recipe_node.parse_recipe_photo_node(_state(""))


# --- SupabaseImageStore -----------------------------------------------------


async def test_supabase_image_store_builds_public_object_url_and_returns_bytes() -> (
    None
):
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, content=_FAKE_IMAGE_BYTES)

    transport = httpx.MockTransport(handler)
    settings = AgentSettings(
        MODEL_NAME="test-model",
        SUPABASE_URL="https://example.supabase.co",
        RECIPE_IMAGE_BUCKET="recipe-images",
    )
    store = SupabaseImageStore(settings=settings, transport=transport)

    result = await store.load("recipes/user-1/photo.jpg")

    assert result == _FAKE_IMAGE_BYTES
    assert requested_urls == [
        "https://example.supabase.co/storage/v1/object/public/"
        "recipe-images/recipes/user-1/photo.jpg"
    ]


async def test_supabase_image_store_raises_when_supabase_url_unset() -> None:
    settings = AgentSettings(MODEL_NAME="test-model", SUPABASE_URL=None)
    store = SupabaseImageStore(settings=settings)

    with pytest.raises(RuntimeError, match="AGENT_SUPABASE_URL"):
        await store.load("recipes/user-1/photo.jpg")


async def test_supabase_image_store_raises_on_non_200_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    settings = AgentSettings(
        MODEL_NAME="test-model", SUPABASE_URL="https://example.supabase.co"
    )
    store = SupabaseImageStore(settings=settings, transport=transport)

    with pytest.raises(RuntimeError, match="404"):
        await store.load("recipes/user-1/missing.jpg")
