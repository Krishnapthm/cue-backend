# Getting started

This tutorial takes you from an empty checkout to one complete chat turn. At
the end, the harness answers a recipe request and shows you an ingredient
checklist.

The tutorial has one path. It makes choices for you. The
[configuration reference](reference/configuration.md) lists the alternatives.

## What you need

- Python 3.13
- [uv](https://docs.astral.sh/uv/), the package manager this repository uses
- A Postgres database, version 14 or later
- A Firebase project, for user sign-in
- An OpenAI API key, for the models the graph calls
- Docker, only if you also run the tests

You do not need a Swiggy account for steps 1 to 6. Step 7 needs one.

## 1. Install the dependencies

```bash
git clone https://github.com/Krishnapthm/cue-backend.git
cd cue-backend
uv sync
```

`uv sync` reads `pyproject.toml`, resolves the versions in `uv.lock`, and
creates the `.venv` directory.

## 2. Write the environment file

```bash
cp .env.example .env
```

Open `.env` and set these four values:

```bash
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/cue
AUTH_FIREBASE_PROJECT_ID=your-firebase-project-id
OPENAI_API_KEY=sk-...
ENVIRONMENT=local
```

Leave the other values as they are. The Swiggy settings are optional. An
endpoint that needs one of them returns HTTP 503 instead of stopping the
service at startup.

Never commit `.env`. The `.gitignore` file already excludes it.

## 3. Create the application tables

```bash
uv run alembic upgrade head
```

Alembic applies the migrations in `migrations/versions/`. These create the
user, chat, cart, order, pantry, tag, and provider tables.

## 4. Create the checkpointer tables

```bash
uv run python scripts/setup_checkpointer.py
```

The agent keeps its own state in separate tables. LangGraph owns them, so
Alembic does not manage them. Run this command once for each database. Run it
again after you change the database, and after a LangGraph upgrade that
changes the checkpoint schema.

## 5. Start the service

```bash
uv run uvicorn main:app --reload
```

The log shows the environment and confirms the graph:

```
Starting cue-api in local
Agent graph compiled and its checkpointer pool opened
```

The graph compiles once for each process, not once for each request.

## 6. Prove that the service works

In a second terminal, call the two probes:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","environment":"local"}

curl http://127.0.0.1:8000/health/db
# {"status":"ok","database":"postgres","server_version":"16.4","latency_ms":3.1}
```

The second probe runs a real query. A 503 response means that the service
cannot reach Postgres.

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The
service serves this page in the `local` and `staging` environments only.

## 7. Get an access token

Every endpoint except `/health` needs a Firebase ID token. The mobile client
supplies one after sign-in. To get one from the command line, call Firebase
directly with a test account and your Firebase Web API key:

```bash
curl -X POST \
  "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=$FIREBASE_WEB_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"...","returnSecureToken":true}'
```

The response holds an `idToken` field. Export it:

```bash
export TOKEN="<idToken>"
```

The harness verifies the signature, the issuer, and the audience against your
Firebase project. It then creates the Cue user row on the first call:

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/auth/me
```

## 8. Link a Swiggy account

The harness buys real groceries, so it needs your Swiggy authorization. Follow
[Link a Swiggy account](guides/link-a-swiggy-account.md), then come back here.

After the link succeeds, this call reports the state:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/providers/swiggy/status
```

## 9. Select a delivery address

Swiggy binds a cart to an address, so a turn cannot run without one. List the
addresses on the linked account:

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/addresses
```

Create a chat session, then attach an address to it:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/chat/sessions
# {"id":"6f2f...","title":null,"selected_address_id":null,"updated_at":"..."}

export SESSION="6f2f..."

curl -X PATCH -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"selected_address_id":"<address id>"}' \
  http://127.0.0.1:8000/chat/sessions/$SESSION
```

If you skip this step, the harness answers the turn with a prompt for an
address. It does not call a model.

## 10. Run one turn

Stream the turn with `curl -N`, which keeps the connection open:

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/chat/sessions/$SESSION/stream?message=I%20want%20to%20cook%20biryani%20for%204"
```

The stream sends named events as the work happens:

```
event: stage
data: {"event":"stage","node":"route_turn"}

event: stage
data: {"event":"stage","node":"generate_recipe"}

event: interrupt
data: {"event":"interrupt","id":"...","payload":{...}}

event: done
data: {"event":"done",...}
```

The turn stops at an `interrupt` event, and the harness waits for your answer.
A turn pauses at most twice:

1. The source choice, when the harness verifies one ready-made component for
   the dish. The payload carries `"ui":"scratch_choice"`.
2. The ingredient checklist, on every recipe turn. The payload carries
   `"ui":"checklist"`.

Read the pending decision at any time:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/chat/sessions/$SESSION/state
```

## 11. Answer the pause

Send the answer as a message with `kind` set to `checklist`. Both cards use
this kind. The harness reads the pending interrupt and applies the matching
schema.

For a source choice, send `choice`:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"role":"user","kind":"checklist","payload":{"choice":"from_scratch"}}' \
  http://127.0.0.1:8000/chat/sessions/$SESSION/messages
```

For the checklist, send `have`, which holds the ingredients you already own:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"role":"user","kind":"checklist","payload":{"have":["salt","oil"]}}' \
  http://127.0.0.1:8000/chat/sessions/$SESSION/messages
```

After the checklist answer, the harness fans out one worker for each remaining
ingredient, composes the cart, and answers with a cart card. The
[agent graph reference](reference/agent-graph.md) describes each event and
each payload.

## What happens next

- Checkout is the next step for the user, and it is not part of the graph. The
  logic lives in `app/cart/checkout.py`, and no HTTP route exposes it yet.
- Read [Architecture](architecture.md) to see why the harness splits the work
  this way.
- Read [Add a graph node](guides/add-a-graph-node.md) to extend the agent.

## If a step fails

| Symptom | Cause | Action |
|---|---|---|
| The service does not start, and the log names `DATABASE_URL` | The setting is absent or invalid | Set `DATABASE_URL` in `.env` |
| `/health/db` returns 503 | Postgres is unreachable | Check the host, the port, and the password encoding |
| A turn fails with an undefined table | The checkpointer tables are absent | Run `scripts/setup_checkpointer.py` |
| Every authenticated call returns 401 | The token is absent, expired, or from another project | Get a new token, and check `AUTH_FIREBASE_PROJECT_ID` |
| A product call returns 503 | The Swiggy settings are absent | See the [configuration reference](reference/configuration.md) |
| A product call returns 401 or 403 | The Swiggy link expired | Link the account again |
