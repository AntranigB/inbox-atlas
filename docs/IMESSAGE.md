# Inbox Atlas on iMessage

Text your inbox. Ask "what do I have today?" or "did I get anything about internships?" and get
a short answer back in the same thread. It also texts you: a morning brief, and a heads up when
new mail lands inside one of your watches.

Built on [Photon Spectrum](https://photon.codes/spectrum) (`spectrum-ts`, open source).

```
 your iPhone  <-- iMessage -->  Photon line  <-- spectrum-ts -->  imessage/ sidecar (:8766)
                                                                     |  POST /api/ask {channel: "imessage"}
                                                                     v
                                                         server.py (:8765) Grok agent + tools
                                                                     |  atlas/notify.py scheduler
                                                                     '--> POST :8766/send (brief, watch alerts)
```

## 1. Photon setup (5 minutes)

1. Sign up at [app.photon.codes](https://app.photon.codes). Use promo code **HACKWITHPHOTON**.
2. Create a project with the iMessage provider. Copy the project ID and project secret.
3. Note the iMessage number (the "line") Photon assigns to your project.
4. Under Dashboard > Users, add your own phone number.

## 2. Env vars (`.env` in the repo root)

| Var | What |
|---|---|
| `SPECTRUM_PROJECT_ID` | Photon project ID |
| `SPECTRUM_PROJECT_SECRET` | Photon project secret |
| `OWNER_PHONE` | your number, E.164 like `+16075551234`. Only this number gets answers. |
| `OWNER_HANDLES` | optional, extra handles that count as you (comma list, e.g. your Apple ID email) |
| `ALLOW_ANY` | `1` answers anyone who texts the line (off by default) |
| `SIDECAR_URL` | where the API server reaches the sidecar, default `http://localhost:8766` |
| `TIMEZONE` | default `America/New_York` |
| `BRIEF_TIME` | morning brief time, default `08:00` |
| `BRIEF` | `0` starts with the brief off (text "brief on" to enable) |
| `QUIET_HOURS` | watch alerts are held in this window and sent after, default `22-7` |
| `POLL_SECONDS` | how often to check Gmail for new mail, default `120` |
| `NOTIFY` | `0` disables the scheduler in the API server |

## 3. Run it

```sh
uv run python server.py            # API server on :8765 (starts the notifier)
cd imessage && npm install && npm start   # sidecar on :8766
```

## 4. Text the line once first (required)

Shared Spectrum lines will not cold start a chat. **Send any message (like "help") from your
phone to the Photon number once.** After that the sidecar remembers the thread, and the
morning brief and watch alerts can reach you. If you skip this, `POST /send` returns an error
that says exactly this, and the API server shows it in `GET /api/notify/status` as `last_error`.

## What you can text

| Text | Does |
|---|---|
| anything | asks the Grok agent over your mail (same tools as the web app), keeps the last 8 turns per thread |
| `today` | what you have today: calendar events plus emails that mention today (`todays_agenda`) |
| `watch internships` | standing region, you get "New email in internships: sender: subject" when new mail matches |
| `watches` / `stop internships` | list / remove watches |
| `brief on` / `brief off` | daily morning brief at `BRIEF_TIME` |
| `reset` | forget the conversation |
| `help` | this list |

Replies are plain text, split at about 600 characters. The sidecar marks your message read and
shows the typing bubble while the agent works.

## Mock mode (no Photon account needed)

```sh
cd imessage && npm run mock        # or: MOCK=1 node src/index.js, or node src/index.js --dry-run
```

A terminal REPL plays your phone: lines you type are iMessages from `OWNER_PHONE`, and anything
the server sends (`POST /api/notify`, the morning brief, watch alerts) prints as an incoming text.
In mock mode the sidecar also accepts `POST :8766/mock/inbound {text, from?}` for scripted demos.

Handy demo calls:

```sh
curl -X POST localhost:8765/api/notify/brief/send     # send the morning brief now
curl -X POST localhost:8765/api/notify/poll           # check for new mail and watch hits now
curl localhost:8765/api/notify/status
curl -X POST localhost:8765/api/notify -H 'content-type: application/json' -d '{"text":"hello"}'
```

## HTTP

Sidecar (`:8766`): `POST /send {text, to?}`, `GET /health`, mock only `POST /mock/inbound`.

API server (`atlas/api/messaging.py`): `POST /api/notify {text, to?}`, `GET /api/notify/status`,
`POST /api/notify/brief {enabled, time?}`, `POST /api/notify/brief/send`, `POST /api/notify/poll`,
`GET /api/notify/agenda?date=`, `GET/POST /api/notify/watches`, `DELETE /api/notify/watches/{name}`.

## Tests

```sh
cd imessage && node --test
ATLAS_ENCODER=hash uv run pytest -q tests/test_notify.py
```
