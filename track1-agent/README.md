# Track 1: Hybrid Token-Efficient Routing Agent

[![Open the live Streamlit demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zeynoamd.streamlit.app/)

**Live demo:** [https://zeynoamd.streamlit.app/](https://zeynoamd.streamlit.app/)

This repository contains our submission for the **AMD Developer Hackathon: ACT II — Track 1**.

The agent minimizes Fireworks token usage with conservative, zero-token local
solvers, task-aware request optimization, response validation, and a single
permitted Fireworks model when a task cannot be answered safely in-container.

## Submission image

The public submission image is:

```text
ghcr.io/zdari23/track1-agent:latest
```

The current image was verified as `linux/amd64` and is approximately 184 MB.
Its published manifest digest is:

```text
sha256:948ae079bee463f424cf095b8e1bba4e5598ad56ac2a50a1745cc070e4b70e06
```

Pull it with:

```bash
docker pull --platform linux/amd64 ghcr.io/zdari23/track1-agent:latest
```

## Evaluation contract

On startup, the container:

1. Reads `/input/tasks.json`.
2. Processes every `{ "task_id", "prompt" }` record.
3. Writes `/output/results.json` as `{ "task_id", "answer" }` records.
4. Exits with status `0` on success and non-zero on an unrecoverable failure.

The official Track 1 harness injects these variables at runtime:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

The submitted image does not contain an API key or `.env` file. All Fireworks
requests use the injected base URL and a model selected from `ALLOWED_MODELS`.

## Run the published image locally

### 1. Prepare input and output directories

Create `input/tasks.json`:

```json
[
  {
    "task_id": "example-01",
    "prompt": "What is the capital of Australia?"
  }
]
```

Ensure the output directory exists:

```bash
mkdir -p output
```

### 2. Configure local credentials

For local development only, create an untracked `.env` file:

```env
FIREWORKS_API_KEY=your_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/kimi-k2p6
```

Use model IDs that are available to your Fireworks account. Never commit
`.env`; the official evaluation harness supplies its own values.

### 3. Run the container

From this directory:

```bash
docker run --rm \
  --platform linux/amd64 \
  --env-file .env \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/zdari23/track1-agent:latest
```

The result will be written to `output/results.json`.

## Build from source

Build an image for the judging architecture:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag track1-agent:latest \
  .
```

Run it with the same mounts and environment variables shown above, replacing
the image name with `track1-agent:latest`.

## Tests

Run the deterministic router tests:

```bash
python3 -m unittest tests.test_router
```

The published image has also been verified for:

- `linux/amd64` architecture
- successful `agent.py` and optimizer imports
- valid `/input/tasks.json` to `/output/results.json` execution
- valid JSON output schema

## Streamlit demo

The hosted Streamlit application is separate from the scored Docker image. It
visualizes the zero-token local routing decision and compares it with a
prompt-based routing baseline before answering with the configured Kimi model.

Streamlit Community Cloud configuration:

- Repository: `zdari23/AMD_Hackathon_RouteMaster`
- Branch: `main`
- Entrypoint: `track1-agent/demo_app.py`
- Custom subdomain: `zeynoamd`

The live demo requires a team-owned Fireworks key in Streamlit **App Settings →
Secrets**:

```toml
FIREWORKS_API_KEY = "your_api_key_here"
```

`MODEL` is optional and defaults to
`accounts/fireworks/models/kimi-k2p6`. Streamlit secrets are used only by the
hosted demo; the official Docker evaluation uses credentials injected by the
hackathon harness.

Do not commit `.env` or `.streamlit/secrets.toml`.
