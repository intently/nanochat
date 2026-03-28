# Connecting to a local NanoChat server

This document explains how another project can connect to a locally running NanoChat web server (the default `scripts.chat_web`) and chat programmatically.

The server exposes a small HTTP API:

- GET  /health           — health check
- GET  /stats            — worker pool and device info
- GET  /logo.svg         — logo image
- GET  /                 — UI (HTML)
- POST /chat/completions — chat API (streaming, Server-Sent Events / text/event-stream)

Default base URL: `http://localhost:8000`

---

**Request format**

POST `/chat/completions`
- Content-Type: `application/json`
- Body JSON shape:
  {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "..."}
    ],
    "temperature": 0.8,   // optional
    "max_tokens": 512,    // optional
    "top_k": 50           // optional
  }

The server streams partial assistant text as Server-Sent Events (SSE) lines. Each SSE "data:" payload is a JSON object. Example payloads:
- {"token": "partial text", "gpu": 0}
- {"done": true}

When you receive `{"done": true}` the response is complete.

---

**Quick curl example (streaming)**

```bash
curl -N -H "Content-Type: application/json" \
  -X POST \
  -d '{"messages":[{"role":"user","content":"Say hello"}]}' \
  http://localhost:8000/chat/completions
```

- `-N` disables buffering so curl prints SSE chunks as they arrive.

---

**Python (synchronous) — using `requests`**

This example reads the SSE stream, extracts `data: ` lines, and parses JSON. It works in a simple script.

```python
import json
import requests

URL = "http://localhost:8000/chat/completions"
payload = {
    "messages": [{"role": "user", "content": "Write a short poem about trees."}],
    "temperature": 0.8,
    "max_tokens": 200,
}

with requests.post(URL, json=payload, stream=True) as r:
    r.raise_for_status()
    assistant_text = []
    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        # SSE lines look like: "data: {...}"
        if raw_line.startswith("data:"):
            data_str = raw_line[len("data:"):].strip()
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if obj.get("done"):
                break
            if "token" in obj:
                # 'token' contains a partial string chunk
                assistant_text.append(obj["token"])
                print(obj["token"], end="", flush=True)

    full = "".join(assistant_text)
    print("\n---\nFull assistant response:\n", full)
```

Notes:
- `requests` is fine for simple scripts. It buffers at the HTTP level unless you use `stream=True` and `iter_lines()`.

---

**Python (async) — using `httpx`**

If your app is async, use `httpx` and `aiter_lines()`.

```python
import asyncio
import json
import httpx

async def main():
    url = "http://localhost:8000/chat/completions"
    payload = {"messages":[{"role":"user","content":"Tell me a joke."}]}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            assistant = []
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data:"):
                    data = raw_line[len("data:"):].strip()
                    obj = json.loads(data)
                    if obj.get("done"):
                        break
                    if "token" in obj:
                        assistant.append(obj["token"])
                        print(obj["token"], end="", flush=True)

    print("\nDONE\nFull: ")
    print("".join(assistant))

if __name__ == "__main__":
    asyncio.run(main())
```

---

**Node / JavaScript (browser)**

On the browser, you can use the native `EventSource` API if CORS allows it. The server sets CORS to allow `*` by default in the repo, so this should work when calling from the same origin.

```javascript
const url = '/chat/completions';

// EventSource only supports GET; the chat endpoint needs POST, so we instead
// use Fetch + read the body as a stream and parse SSE ourselves.

async function streamChat() {
  const payload = { messages: [{ role: 'user', content: 'Hello from browser' }] };
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const chunk = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 2);
      // chunk contains lines like "data: {...}\n"
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data:')) {
          const j = JSON.parse(line.slice('data:'.length).trim());
          if (j.done) { console.log('done'); return; }
          if (j.token) {
            // append to UI
            console.log('partial:', j.token);
          }
        }
      }
    }
  }
}

streamChat();
```

**Node (server-side) — using `eventsource-parser` or a streaming fetch**

See libraries like `eventsource-parser` or use `node-fetch`/`undici` to stream the response and parse `data:` lines similarly to the browser example.

---

**Health & status checks**
- `GET /health` — returns JSON `{status: "ok", ready: true|false, num_gpus: X, available_workers: Y}`
- `GET /stats` — returns worker-level device info (device strings like `cuda:0` or `cpu`). Use this to determine if the server loaded models on GPU.

Note: `/stats` now also includes `model_tag` and `model_step` when a checkpoint was selected at startup. These identify which checkpoint directory (e.g. `d32`) and step (e.g. `000650`) the server loaded.

Example:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

---

**Common gotchas**
- Make sure you are calling the correct origin and port (default `localhost:8000`).
- If the server was started with `--device-type cpu`, it will run on CPU even if CUDA is available.
- If you get `FileNotFoundError` on startup, the server couldn't find model checkpoints. Either place checkpoints in the expected base dir (default: `%USERPROFILE%/.cache/nanochat/`) or start the server in UI-only mode.
- Keep `stream=True` (`requests`) or `stream`/`client.stream` (`httpx`) — otherwise you will block until the entire stream completes and miss incremental tokens.

---

If you'd like, I can add a minimal helper client module in this repo that wraps the HTTP streaming and emits full/partial events for easier reuse.