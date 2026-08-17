# Link a Swiggy account

The harness buys real groceries, so each user authorizes Cue against their own
Swiggy account. This guide completes the OAuth 2.1 and PKCE flow.

Read this if you set up a development machine, or if you build a client. For
the endpoint contracts, see the
[HTTP API reference](../reference/api.md#providers-the-swiggy-link).

## Before you start

Set the four Swiggy variables in `.env`:

```bash
SWIGGY_CLIENT_ID=your-swiggy-client-id
SWIGGY_REDIRECT_URI=https://api.example.com/providers/swiggy/callback
SWIGGY_APP_CALLBACK_DEEP_LINK=cue://swiggy-link
SWIGGY_TOKEN_ENCRYPTION_KEY=your-fernet-key
```

Generate the encryption key once:

```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Register `SWIGGY_REDIRECT_URI` with Swiggy. The value must match exactly on
the authorize call and on the token call. Read
[the Swiggy authentication page](https://mcp.swiggy.com/builders/docs/start/authenticate)
for the current rules, or
[the Markdown version](https://mcp.swiggy.com/builders/docs/start/authenticate.md)
if an agent reads it.

Without these four values, every provider endpoint returns 503. The service
still starts, so local work that never touches Swiggy is unaffected.

## Choose a path

The flow has two endings, and the client decides which one it uses.

| Path | Use it when | Ending |
|---|---|---|
| Redirect | The redirect URI reaches this server, for example on a deployed host | `GET /providers/swiggy/callback` |
| Intercepted | The redirect URI cannot reach this server, for example a `localhost` URI on a physical phone | `POST /providers/swiggy/callback` |

The mobile app uses the intercepted path. A `localhost` redirect URI resolves
to the phone itself, so the browser redirect never arrives at the server.

## Step 1. Start the flow

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/providers/swiggy/authorize
```

```json
{
  "authorize_url": "https://mcp.swiggy.com/oauth/authorize?...",
  "redirect_uri": "https://api.example.com/providers/swiggy/callback"
}
```

The service records an OAuth transaction. It holds the PKCE verifier
encrypted at rest, and the `state` value that binds the transaction to this
Cue user.

## Step 2. Let the user consent

Open `authorize_url` in a browser, or in a WebView. The user signs in to
Swiggy and approves the access.

## Step 3a. Finish through the redirect

Swiggy redirects the browser to your registered URI with `code` and `state`.
The service exchanges the code, stores the encrypted access token, and then
redirects to your deep link:

```
cue://swiggy-link?swiggy_link=success
cue://swiggy-link?swiggy_link=error
```

The redirect happens for success and for failure alike, so the app can resume
whatever action the user started.

This endpoint takes no bearer token, because a browser calls it. It therefore
cannot check the Cue user, and it relies on the `state` value alone.

## Step 3b. Finish through the intercepted redirect

The client runs the consent page in a WebView and watches the navigation. When
the WebView tries to open the redirect URI, the client cancels that navigation
and reads `code` and `state` from the URL. It then posts them over its normal
authenticated connection:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"code":"<code>","state":"<state>"}' \
  http://127.0.0.1:8000/providers/swiggy/callback
```

```json
{"status": "connected"}
```

This path is stricter than the redirect path. It knows the Cue user, so it
rejects a `state` that belongs to somebody else.

| Status | Cause |
|---|---|
| 400 | The state is unknown, expired, already used, or another user's |
| 502 | Swiggy rejected or failed the code exchange |

## Step 4. Check the link

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/providers/swiggy/status
```

| `status` | Meaning | What the client does |
|---|---|---|
| `connected` | The link works | Nothing |
| `reconnect_needed` | The token expired or Swiggy rejected it | Start this flow again |
| `not_connected` | The user never linked an account | Offer the link |

The harness reports the same condition during a turn. A dead token ends the
turn with an `error` event whose `action` is `reconnect_swiggy`.

## Unlink

```bash
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/providers/swiggy
```

The service answers 204 and forgets the stored token.

## Key rotation

`SWIGGY_TOKEN_ENCRYPTION_KEY` encrypts the access tokens and the PKCE
verifiers at rest. A new key makes every stored value unreadable, so every
user must link the account again. Rotate the key only when you accept that
cost.
