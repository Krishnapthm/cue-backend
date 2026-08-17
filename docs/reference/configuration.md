# Configuration reference

Every setting is an environment variable. The service reads `.env` in local
development, and reads the real process environment everywhere else. A real
environment variable always wins over a `.env` value.

Copy `.env.example` to `.env` to start. Never commit `.env`.

## How settings load

Each domain owns one `BaseSettings` class with its own prefix. There is no
single global settings object.

| Class | Module | Prefix |
|---|---|---|
| `Settings` | `app/config.py` | none |
| `AuthSettings` | `app/auth/config.py` | `AUTH_` |
| `ProviderSettings` | `app/providers/config.py` | `SWIGGY_` |
| `AgentSettings` | `app/agent/config.py` | `AGENT_` |

The classes parse `.env` into their own typed fields. They never export it to
the process environment. Libraries that read `os.getenv` directly, which
includes the OpenAI, Anthropic, and LangSmith SDKs, would therefore see
nothing. `main.py` calls `load_dotenv()` before it imports any application
module to close that gap. `load_dotenv()` does not overwrite a real
environment variable, which matches Pydantic's own order.

A setting with no default and no value stops the service at startup. This is
deliberate for `DATABASE_URL` and `AUTH_FIREBASE_PROJECT_ID`, because no
request can succeed without them. The Swiggy settings are optional instead,
and an endpoint that needs one returns 503.

## Core

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Postgres DSN | none, required | The application database. Use the `postgresql+asyncpg://` scheme |
| `ENVIRONMENT` | `local`, `staging`, `production` | `local` | `production` disables `/openapi.json` and `/docs` |
| `CORS_ALLOW_ORIGINS` | JSON array of strings | `[]` | Browser origins that may call the API |
| `DATABASE_POOL_PRE_PING` | boolean | `true` | Verify a connection before the pool hands it out |
| `DATABASE_POOL_SIZE` | integer | `5` | SQLAlchemy pool size |
| `DATABASE_MAX_OVERFLOW` | integer | `10` | Connections above the pool size |
| `DATABASE_ECHO` | boolean | `false` | Log every SQL statement |

Notes:

- URL-encode the database password when it contains `@ : / ? # [ ] %`.
- `CORS_ALLOW_ORIGINS` defaults to empty, so an unconfigured deployment
  permits nothing. Never set a wildcard. The client sends a bearer token on
  every call, and the origin list is the only control on who may do that from
  a browser. Native iOS and Android do not apply CORS. Expo's web target does.
- Keep `DATABASE_POOL_PRE_PING` on with a hosted Postgres. Supabase closes
  idle connections on the server, and the first query after an idle period
  fails without the ping.

## Authentication

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AUTH_FIREBASE_PROJECT_ID` | string | none, required | The Firebase project. The service verifies each token's `aud` and `iss` against it |

## Swiggy

All four are optional, so local work that never links an account still runs.
An endpoint that needs one raises `ProviderNotConfiguredError`, and the
service answers 503.

| Variable | Type | Purpose |
|---|---|---|
| `SWIGGY_CLIENT_ID` | string | The OAuth 2.1 client id from Swiggy |
| `SWIGGY_REDIRECT_URI` | URL | The callback the service exposes. Swiggy must have it registered, and it must match on both the authorize call and the token call |
| `SWIGGY_APP_CALLBACK_DEEP_LINK` | URL | The app deep link the callback redirects to, on success and on failure |
| `SWIGGY_TOKEN_ENCRYPTION_KEY` | Fernet key | Encrypts the access token and the PKCE verifier at rest |

Generate the encryption key once:

```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Rotating this key makes every stored token unreadable. Each user must link
the account again.

## Agent

Model choice is configuration, never code. No node names a model. A node asks
for a role, and these settings say which model serves it.

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AGENT_MODEL_PROVIDER` | `openai` or `anthropic` | `openai` | Which client `providers.get_chat_model` returns |
| `AGENT_MODEL_ROUTER` | string | `gpt-5.4-nano-2026-03-17` | Turn classification, on every turn |
| `AGENT_MODEL_RECIPE` | string | `gpt-5.6-luna` | Recipe generation |
| `AGENT_MODEL_VISION` | string | `gpt-5.6-luna` | Recipe photo parsing |
| `AGENT_MODEL_ORDER_STATUS` | string | `gpt-5.4-nano-2026-03-17` | One status sentence |
| `AGENT_MODEL_TITLE` | string | `gpt-4o-mini` | The session name |
| `AGENT_MODEL_ROUTER_REASONING_EFFORT` | effort | `none` | Reasoning effort for the router |
| `AGENT_MODEL_ORDER_STATUS_REASONING_EFFORT` | effort | `none` | Reasoning effort for the status node |
| `AGENT_CHECKPOINTER_POOL_MIN_SIZE` | integer | `2` | Connections the checkpointer keeps warm |
| `AGENT_CHECKPOINTER_POOL_MAX_SIZE` | integer | `10` | Checkpointer pool ceiling |
| `AGENT_SUPABASE_URL` | URL | none | Supabase project base URL, for recipe photo object URLs |
| `AGENT_RECIPE_IMAGE_BUCKET` | string | `recipe-images` | The Storage bucket that photo uploads land in |

Effort values: `none`, `low`, `medium`, `high`, `xhigh`.

Notes:

- The defaults are a costed decision, not a deployment preference. The router
  and the status node run a cheap model, because they label and rephrase. The
  recipe and vision roles run the strongest model, because a wrong ingredient
  becomes a wrong cart and then a wrong order.
- Pass `reasoning_effort` only to a model that offers the dial. `gpt-4o-mini`
  rejects the argument with HTTP 400, which is why the title role sends none.
- The checkpointer pool is separate from the SQLAlchemy pool. The checkpointer
  speaks psycopg, and the application speaks asyncpg. Both count against the
  same Postgres `max_connections`, so size them together.
- The checkpointer pool is sized by checkpoint operations in flight, not by
  open streams. A pooled connection is borrowed for one read or write, then
  returned, so an idle stream holds nothing.
- `AGENT_SUPABASE_URL` is optional. `SupabaseImageStore.load` raises a clear
  error when a photo turn needs it and it is absent.

## Model provider keys

The langchain clients read these directly from the process environment. They
are not settings fields.

| Variable | Needed when |
|---|---|
| `OPENAI_API_KEY` | `AGENT_MODEL_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `AGENT_MODEL_PROVIDER=anthropic` |

The test suite removes both. A model call that escapes its stub then fails
loudly instead of billing a live request.

## Tracing

The LangSmith SDK reads these directly from the process environment.

| Variable | Purpose |
|---|---|
| `LANGSMITH_TRACING` | Set to `true` to trace |
| `LANGSMITH_API_KEY` | The LangSmith key. Leave it unset to run with tracing off |
| `LANGSMITH_PROJECT` | The project that receives the traces |
| `LANGSMITH_ENDPOINT` | The API host. Defaults to `https://api.smith.langchain.com` |

## Local example

```bash
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/cue
ENVIRONMENT=local
CORS_ALLOW_ORIGINS=["http://localhost:8081"]
AUTH_FIREBASE_PROJECT_ID=cue-dev
AGENT_MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=cue-agent
```

This configuration runs everything except the Swiggy paths. Add the four
`SWIGGY_` values to link an account.
