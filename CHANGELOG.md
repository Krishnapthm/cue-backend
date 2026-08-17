# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The project has no tagged release yet. Everything below is unreleased, and it
describes the state of `main` since the first commit.

## [Unreleased]

### Added

- **The agent harness.** A LangGraph state graph turns a dish name into a
  verified Swiggy Instamart cart. The graph owns the order of the work, and
  the model does the language work only.
- **A four-way entry router.** Every turn is classified as a recipe, a photo,
  an order-status question, or off-topic. An off-topic turn is refused before
  any recipe model call. The classifier treats the user's text as data to
  judge, never as instructions to follow.
- **Recipe generation, from text and from a photo.** Both paths produce the
  same structured recipe, and the photo path joins the text path afterwards.
- **A ready-made source choice.** The harness verifies one ready-made
  component against the selected address, then asks the user whether to buy it
  or to cook it. It asks only when a verified alternative exists.
- **The ingredient checklist.** The harness shows have and need rows, and
  pauses. In-stock pantry items arrive already ticked, and the user's own
  answer always wins over that seed.
- **A parallel ingredient fan-out.** One worker for each needed ingredient
  searches Instamart at the same time. Each worker streams its result as it
  finishes.
- **Cart composition and a closing card.** The harness records a plan, pushes
  the lines to Swiggy, and renders the card from the cart Swiggy holds,
  including the minimum-order verdict.
- **Order status in chat.** A "where is my order" turn is answered from the
  user's real order list, in one sentence.
- **Automatic session names.** A session is named after the resolved dish,
  once, in the background. A failure never affects the turn.
- **Streaming turns.** `GET /chat/sessions/{id}/stream` sends typed
  server-sent events: `token`, `match`, `stage`, `interrupt`, `error`, and one
  final `done`. `GET /chat/sessions/{id}/state` returns the pending decision,
  so a client that reconnects days later finds it again.
- **Swiggy account linking.** OAuth 2.1 and PKCE, with two endings: a browser
  redirect, and an intercepted redirect that a mobile client posts back.
  Access tokens and PKCE verifiers are encrypted at rest.
- **Instamart tool wrappers.** Addresses, product search, cart read and
  mutation, checkout, order history, order details, go-to items, and order
  tracking.
- **Variant selection and substitution.** The harness picks the pack size,
  computes the quantity, and ranks substitutes by pack-size distance and by
  the brands the user already buys. Every substitution is user-visible.
- **Checkout safety.** `place_order` refuses an empty cart, refuses a second
  concurrent checkout, and never retries a transport failure. It records the
  order as `unknown` and reads Swiggy's recent orders instead, because Swiggy
  offers no idempotency key.
- **A per-user pantry.** Fixed server-side categories in display order, and a
  0 to 3 stock level. The pantry seeds the checklist.
- **NFC tag binding.** A tap resolves a tag to an orderable variant, single or
  batched, with alternates. A batch is capped at 50 taps and 5 concurrent
  searches.
- **Firebase authentication.** The service verifies the ID token, caches
  Google's key set, and creates the Cue user row with one upsert.
- **Health probes.** `/health` for liveness, and `/health/db` for a real
  query.
- **LangSmith tracing**, and `scripts/agent_smoke.py`, which proves that a run
  is queryable rather than merely visible.
- **An MIT license**, in `LICENSE`.
- **Code owners**, in `.github/CODEOWNERS`, so GitHub requests a review on
  every pull request.
- **Documentation.** A README that frames the repository as an agent harness,
  a getting-started tutorial, an architecture explanation, three reference
  pages, two how-to guides, and this changelog.

### Changed

- **The compiled graph is now scoped to the application lifespan.** It was
  compiled for each request, which opened one dedicated database connection
  for each call. A long-lived stream then pinned that connection for the whole
  turn, and a few concurrent users exhausted the database. The graph now
  compiles once for each process, backed by a connection pool that lends a
  connection for one checkpoint read or write.
- **Model choice moved from one setting to five roles.** `AGENT_MODEL_NAME` is
  replaced by `AGENT_MODEL_ROUTER`, `AGENT_MODEL_RECIPE`, `AGENT_MODEL_VISION`,
  `AGENT_MODEL_ORDER_STATUS`, and `AGENT_MODEL_TITLE`. Each carries a costed
  default, so a deployment overrides one only on purpose.
- **The scope guardrail became the entry router.** One node now answers both
  questions, in-scope and which branch, instead of two classifiers in
  sequence.
- **Checkout left the graph**, so no operation that spends money sits behind a
  model decision.
- **Session titles use a small OpenAI model.** The previous model rejected the
  `reasoning_effort` argument with HTTP 400.

### Fixed

- **The Instamart response envelope.** The product schema read `name` and
  `variants` while Swiggy sends `displayName` and `variations`. Every tag
  resolution returned `unresolved`, and the tests stayed green.
  `scripts/capture_search_products.py` now reads a real payload, so the next
  field is confirmed rather than guessed.
- **Product ratings** are preserved as the strings Swiggy sends.
- **Assumed staples** no longer appear on the ingredient checklist.
- **The selected delivery address** is persisted on the chat session, so a
  turn no longer loses it.
- **A relabelled NFC sticker** invalidates its binding, instead of resolving
  to the previous product.
- **A cached tag resolution** always offers the bound variant, and always
  searches, so the alternates are never hidden.
- **LangSmith tracing** reaches the SDK. The settings classes parse `.env`
  into typed fields and never export it, so `main.py` now loads the file into
  the process environment before it imports any application module.
- **The Swiggy token response** is accepted when it omits `scope`.
- **The OpenAPI Authorize button** works, because the service declares a
  bearer security scheme.
- **Swiggy OAuth settings are optional**, so local development that never
  links an account still starts.
