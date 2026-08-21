# HTTP API reference

This page lists every endpoint the service exposes. The source of truth is the
OpenAPI schema at `/openapi.json`, which the service serves in the `local` and
`staging` environments. The `production` environment sets `openapi_url=None`,
so it serves neither the schema nor `/docs`.

Base URL in local development: `http://127.0.0.1:8000`.

## Authentication

Every endpoint except `/health`, `/health/db`, and `GET /providers/swiggy/callback`
needs a Firebase ID token:

```
Authorization: Bearer <firebase id token>
```

The service verifies the signature against Google's public keys, and verifies
the issuer and the audience against `AUTH_FIREBASE_PROJECT_ID`. It caches the
key set for a fixed period.

The first call for a new Firebase account creates the Cue user row. The
service uses one upsert, never a select and then an insert, so two concurrent
first calls cannot create two rows.

An absent, malformed, or expired token returns 401.

## Errors

Domain exceptions subclass `AppError`. A global handler renders each one as:

```json
{"detail": "Resource not found."}
```

| Status | Meaning |
|---|---|
| 400 | The request is malformed at the domain level, for example an unknown OAuth state |
| 401 | The Firebase token failed, or the Swiggy session expired |
| 404 | The resource does not exist, or another user owns it |
| 409 | The request conflicts with the current state |
| 422 | The request body failed validation, or Swiggy rejected it |
| 502 | An upstream service failed, for example Swiggy or the model provider |
| 503 | A dependency is unreachable or unconfigured, for example the database or the Swiggy client settings |

The service answers 404, not 403, for a resource that another user owns. This
keeps the existence of another user's data private.

Several request schemas accept both `snake_case` and `camelCase` field names,
because `populate_by_name` is set. The tables below print the JSON name the
mobile client sends.

## Health

### GET /health

Reports that the process is up. Does not touch the database.

```json
{"status": "ok", "environment": "local"}
```

### GET /health/db

Runs a real query. Returns 503 when Postgres is unreachable.

```json
{"status": "ok", "database": "postgres", "server_version": "16.4", "latency_ms": 3.1}
```

## Auth

### GET /auth/me

Returns the signed-in Cue user, and creates the row on the first call.

| Field | Type |
|---|---|
| `id` | integer |
| `email` | string |
| `display_name` | string or null |

## Providers, the Swiggy link

All paths use the prefix `/providers/swiggy`. See
[Link a Swiggy account](../guides/link-a-swiggy-account.md) for the procedure.

### POST /providers/swiggy/authorize

Starts the OAuth 2.1 and PKCE flow. Returns the URL to open, and the redirect
URI the service registered.

| Field | Type |
|---|---|
| `authorize_url` | string |
| `redirect_uri` | string |

### GET /providers/swiggy/callback

The redirect target Swiggy calls. Takes the `code`, `state`, and `error` query
parameters. Answers with a 307 redirect to `SWIGGY_APP_CALLBACK_DEEP_LINK`,
for success and for failure alike, so the app can continue the pending action.
This endpoint needs no bearer token, because a browser calls it.

### POST /providers/swiggy/callback

Completes the link when the app intercepts the redirect itself.

Request:

| Field | Type | Rule |
|---|---|---|
| `code` | string | At least 1 character |
| `state` | string | At least 1 character |

Returns the same body as `GET /providers/swiggy/status`.

| Status | Cause |
|---|---|
| 400 | The state is unknown, expired, already used, or owned by another user |
| 502 | Swiggy rejected or failed the code exchange |

### GET /providers/swiggy/status

| Field | Values |
|---|---|
| `status` | `connected`, `reconnect_needed`, `not_connected` |

### DELETE /providers/swiggy

Unlinks the account. Returns 204.

## Addresses

Addresses live on the Swiggy account, not in the Cue database.

### GET /addresses

Returns the saved delivery addresses.

| Field | Type |
|---|---|
| `id` | string |
| `addressLine` | string |
| `phoneNumber` | string or null |
| `addressCategory` | `HOME`, `WORK`, `OFFICE`, `FRIENDS_AND_FAMILY`, `OTHER`, or null |
| `addressTag` | string or null |

### POST /addresses

Creates an address on the Swiggy account. Returns 201.

| Field | Type | Required |
|---|---|---|
| `fullAddress` | string | Yes |
| `addressLine` | string | Yes |
| `addressLine2` | string | No |
| `city` | string | Yes |
| `postalCode` | string | Yes |
| `latitude` | number | Yes |
| `longitude` | number | Yes |
| `addressCategory` | enum, see above | Yes |
| `userName` | string | Yes |
| `userPhone` | string | Yes |
| `locality` | string | No |
| `addressTag` | string | No |
| `receiverName` | string | No |
| `receiverPhone` | string | No |

Returns 422 when Swiggy rejects the address.

### DELETE /addresses/{address_id}

Returns 204. Returns 422 when no such address exists on the linked account.

## Chat

All paths use the prefix `/chat/sessions`. This is the harness surface.

### POST /chat/sessions

Creates a new, untitled session. Returns 201.

| Field | Type |
|---|---|
| `id` | UUID |
| `title` | string or null |
| `selected_address_id` | string or null |
| `updated_at` | datetime |

The session id is also the LangGraph `thread_id`. The harness names the
session after the dish once a recipe resolves, so `title` fills in later.

### GET /chat/sessions

Lists the caller's sessions, most recently updated first.

### PATCH /chat/sessions/{session_id}

Selects the delivery address for the session. A turn cannot run without one.

| Field | Type | Rule |
|---|---|---|
| `selected_address_id` | string | 1 to 100 characters |

### GET /chat/sessions/{session_id}

Returns the session and its ordered transcript.

Each message carries:

| Field | Type |
|---|---|
| `id` | integer |
| `role` | `user` or `assistant` |
| `kind` | `text`, `image`, `checklist`, `cart_ready` |
| `content` | string or null |
| `payload` | object or null |
| `created_at` | datetime |

A `text` message carries `content`. Every other kind carries `payload`.

### GET /chat/sessions/{session_id}/stream

Runs one turn and streams it as server-sent events. This is the endpoint the
mobile client uses.

| Parameter | In | Rule |
|---|---|---|
| `message` | query | At least 1 character |

The method is GET, because a browser `EventSource` can issue only a GET.

The service checks ownership before the response starts, so an unauthorized
request still returns 404. After the first byte, the status code is settled,
and any later failure arrives as an `error` event.

The response carries the media type `text/event-stream`. See
[Agent graph reference](agent-graph.md#stream-events) for every event shape.

### GET /chat/sessions/{session_id}/state

Returns the decision the agent waits on, or null when the session is idle.

```json
{"pending_interrupt": {"id": "...", "payload": {"ui": "checklist", "items": []}}}
```

A client calls this after a reconnect, or on a cold start days later. The
service reads the checkpointer, so there is no second source of truth.

### POST /chat/sessions/{session_id}/messages

Appends a message and runs the agent on it. Returns 201, and returns the whole
turn in one payload.

| Field | Type | Rule |
|---|---|---|
| `role` | `user` or `assistant` | Required |
| `kind` | message kind | Defaults to `text` |
| `content` | string or null | Required when `kind` is `text` |
| `payload` | object or null | Required when `kind` is not `text` |

Response:

| Field | Type |
|---|---|
| `user_message` | message |
| `assistant_message` | message or null |

Only a user's `text` message starts a recipe turn. A `checklist` message
answers a pause, and its payload must carry `have` or `choice`. Every other
message persists and returns `assistant_message: null`.

| Status | Cause |
|---|---|
| 404 | No such session for this user |
| 422 | The body does not match its declared kind, or no pause is open |
| 502 | The agent could not produce a reply |

## Cart

All paths use the prefix `/cart`. The cart lives on the Swiggy server. These
endpoints read and mutate it directly, outside the graph.

### GET /cart

Returns the current Swiggy cart.

| Field | Type |
|---|---|
| `items` | list of lines: `spinId`, `quantity`, `price`, `productName`, `imageUrl`, `rating` |
| `total` | number or null |
| `minimumOrderValue` | number or null |
| `availablePaymentMethods` | list of strings |

### POST /cart/items

Adds items, and keeps what the cart already holds.

| Field | Type | Rule |
|---|---|---|
| `addressId` | string | At least 1 character |
| `items` | list | At least 1 entry |
| `items[].spinId` | string | At least 1 character |
| `items[].quantity` | integer | Greater than 0 |

Response:

| Field | Type |
|---|---|
| `cart` | the cart after the change |
| `added` | the lines Swiggy accepted |
| `rejected` | the lines Swiggy refused, each with a `reason` |

### PATCH /cart/items/{spin_id}

Sets one line's quantity.

| Field | Type | Rule |
|---|---|---|
| `addressId` | string | At least 1 character |
| `quantity` | integer | Greater than 0. Use DELETE to remove a line |

### DELETE /cart/items/{spin_id}

Removes one line. Takes `address_id` as a query parameter. Returns the same
mutation result as the other cart endpoints.

### DELETE /cart

Removes every line, in one write. Takes `address_id` as a query parameter.
Instamart has no separate clear-cart tool; this sends `update_cart` with an
empty item list, which is how it expresses an emptied cart. Unlike the other
mutating routes this does not merge - clearing is the one operation meant to
discard everything.

## Orders

All paths use the prefix `/orders`.

### GET /orders

Returns recent Instamart orders.

| Field | Type |
|---|---|
| `order_id` | string |
| `status` | `preparing`, `out_for_delivery`, `delivered`, `cancelled` |
| `placed_at` | string or null |
| `items` | list of strings |
| `total` | number or null |

### GET /orders/{order_id}

Returns the line items and the bill breakdown: `item_total`, `delivery_fee`,
`handling_fee`, and `grand_total`. Returns 422 when the order is not the
caller's.

### GET /orders/{order_id}/track

Returns live tracking. Takes `lat` and `lng` as query parameters.

| Field | Type |
|---|---|
| `order_id` | string |
| `status` | `active` or `delivered` |
| `eta` | string or null |
| `delivery_partner_location` | object with `lat` and `lng`, or null |

The service throttles this call, so a client that polls faster than the floor
receives the cached answer. Returns 422 when Swiggy cannot track the order.

## Pantry

All paths use the prefix `/pantry`. The pantry seeds the checklist: an item
that is in stock arrives already ticked.

### GET /pantry

Returns the items in category display order. The server owns that order.

| Field | Type |
|---|---|
| `id` | integer |
| `name` | string |
| `category` | see the category table |
| `level` | integer, 0 to 3 |
| `last_bought_at` | datetime or null |

Categories, in display order: `Grains & pulses`, `Spices & masalas`,
`Vegetables & fruit`, `Dairy & eggs`, `Oils & condiments`,
`Snacks & packaged`.

`level` is an ordinal, not a percentage: 0 is out, 1 is low, 2 is half, and 3
is full. A new item defaults to 3.

### POST /pantry

Adds an item, or updates the item that already uses that name. Returns 201.

| Field | Type | Rule |
|---|---|---|
| `name` | string | Up to 200 characters |
| `category` | category | Required |
| `level` | integer | 0 to 3, defaults to 3 |

### PATCH /pantry/{item_id}

Updates a part of an item, in practice the level. Returns 409 when the new
name belongs to another item.

### DELETE /pantry/{item_id}

Returns 204.

## Tags, NFC

All paths use the prefix `/pantry/tags`. A user sticks an NFC tag on a
container. A tap resolves the tag text to an orderable Instamart variant, and
the service remembers the binding.

### POST /pantry/tags/resolve

Resolves one tap, and returns the alternates beside it.

| Field | Type | Rule |
|---|---|---|
| `addressId` | string | Required |
| `tagUid` | string | Required |
| `text` | string | Required |
| `quantity` | integer | Defaults to 1 |

Response: one resolution, plus `candidates`.

| Field | Type |
|---|---|
| `tag_uid` | string |
| `text` | string |
| `outcome` | `cached`, `bound`, `unresolved` |
| `spin_id` | string or null |
| `product_id` | string or null |
| `product_name` | string or null |
| `refill_size` | string or null |
| `unit_price` | number or null |
| `in_stock` | boolean or null |
| `pantry_item_id` | integer or null |
| `quantity` | integer |
| `candidates` | list of alternates |

### POST /pantry/tags/resolve-batch

Resolves a finished scan.

| Field | Type | Rule |
|---|---|---|
| `addressId` | string | Required |
| `taps` | list | 1 to 50 entries |

The service runs at most 5 searches at once, so one scan cannot spend the
whole Swiggy quota. An `unresolved` outcome is a normal entry inside a 200
response. One slug with no product must not fail the other nine.

### PATCH /pantry/tags/{tag_uid}

Binds the tag to a different variant.

| Field | Type | Rule |
|---|---|---|
| `spinId` | string | Required |
| `addressId` | string | Required |
| `productId` | string | Optional |
| `productName` | string | Optional |
| `refillSize` | string | Optional |
| `unitPrice` | number | Optional, 0 or more |

### DELETE /pantry/tags/{tag_uid}

Unbinds the tag. Returns 204.
