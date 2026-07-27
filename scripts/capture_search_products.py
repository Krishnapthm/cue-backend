"""Capture a full, untrimmed `search_products` response (CUE-78).

Swiggy documents only the `{success, data, message}` envelope for
`search_products` - the fields inside `data` are not specified anywhere, so
the only way to answer "does a product carry an image URL or a rating?" is to
look at the wire. CUE-77 is the cautionary tale for guessing instead: `Product`
read `name`/`variants` while Swiggy sent `displayName`/`variations`, 400+ tests
stayed green, and every tag resolution in production returned `unresolved`.

Run it against a linked account, with `DATABASE_URL` and
`SWIGGY_TOKEN_ENCRYPTION_KEY` set in `.env` as usual:

    uv run python scripts/capture_search_products.py --user-id 1 --address-id <id>

It prints the per-product and per-variation key sets - which is what belongs
in the issue - and writes the raw payload to a local file for inspection. The
payload is written to disk rather than logged, because a real response is user
data; keep the file out of git and delete it when the question is answered.

Pass `--list-addresses` to look up the account's address ids first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from app.database import session_factory  # noqa: E402
from app.instamart import client, service  # noqa: E402
from app.instamart.constants import TOOL_SEARCH_PRODUCTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("capture_search_products")

DEFAULT_OUTPUT = Path("search_products_capture.json")


def _key_union(records: list[dict[str, Any]]) -> list[str]:
    """Every key seen across a list of records, so a sparse field is not missed.

    A field Swiggy only populates for some products - which is exactly what an
    optional image or rating would look like - would be invisible if we only
    read the first record.
    """
    keys: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            keys.update(record)
    return sorted(keys)


def _report(data: Any) -> None:
    """Print the per-product and per-variation key sets, and flag the question."""
    if not isinstance(data, dict):
        logger.warning("`data` was %s, not an object: %r", type(data).__name__, data)
        return

    logger.info("data keys: %s", sorted(data))
    products = data.get("products") or []
    if not products:
        logger.warning("No products in the response - try a broader query.")
        return

    product_keys = _key_union(products)
    variations = [
        variation
        for product in products
        if isinstance(product, dict)
        for variation in (product.get("variations") or [])
    ]
    variation_keys = _key_union(variations)

    logger.info("%d products, %d variations", len(products), len(variations))
    logger.info("per-product keys:   %s", product_keys)
    logger.info("per-variation keys: %s", variation_keys)

    # The two fields CUE-78 exists to settle. Reported as "candidates worth
    # looking at", never as a conclusion - the key set above is the answer,
    # and a field named something unexpected is exactly what CUE-77 was.
    for label, needles in (
        ("image", ("image", "img", "photo", "thumb", "cloudinary", "media")),
        ("rating", ("rating", "review", "star")),
    ):
        hits = sorted(
            key
            for key in set(product_keys) | set(variation_keys)
            if any(needle in key.lower() for needle in needles)
        )
        logger.info("possible %s fields: %s", label, hits or "NONE FOUND")


async def main() -> None:
    """Make one real `search_products` call and report what came back."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--address-id", default=None)
    parser.add_argument("--query", default="sugar")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--list-addresses",
        action="store_true",
        help="Print the account's address ids and exit.",
    )
    args = parser.parse_args()

    async with session_factory() as session:
        if args.list_addresses:
            for address in await service.get_addresses(session, args.user_id):
                logger.info("%s  %s", address.id, address.address_line)
            return

        if args.address_id is None:
            parser.error("--address-id is required (use --list-addresses to find one)")

        token = await service.resolve_access_token(session, args.user_id)

    # Called at the client level, not through `service.search_products`, so the
    # payload is captured before `Product` parsing drops every field the schema
    # does not yet know about - which is the entire point.
    data = await client.call_tool(
        token,
        TOOL_SEARCH_PRODUCTS,
        {"addressId": args.address_id, "query": args.query, "offset": 0},
    )

    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Raw payload written to %s", args.output)
    _report(data)


if __name__ == "__main__":
    asyncio.run(main())
