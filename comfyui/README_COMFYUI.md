# ComfyUI Integration

This project exposes a local OpenAI-compatible API so ComfyUI can talk to your fine-tuned physics model through any custom node that supports OpenAI-style chat completions.

## 1. Install ComfyUI

Follow the normal ComfyUI installation guide for your platform.

## 2. Install An API/LLM Custom Node

Use any custom node that can call an OpenAI-compatible endpoint. Common examples include:

- ComfyUI LLM Party
- OpenAI-compatible chat/completions nodes
- Any HTTP/API node that can send JSON to `POST /v1/chat/completions`

Node names and widget layouts vary, so this repo provides a placeholder workflow JSON rather than a hardcoded vendor-specific export.

## 3. Start The Local API

From the project root:

```bash
python scripts/06_serve_openai_api.py --adapter outputs/adapters/final
```

If you merged the model first:

```bash
python scripts/06_serve_openai_api.py --merged-model outputs/merged_model
```

Default server URL:

```text
http://127.0.0.1:8000/v1
```

Health check:

```text
http://127.0.0.1:8000/health
```

## 4. Configure The ComfyUI Node

Set:

- Base URL: `http://127.0.0.1:8000/v1`
- Model name: `physics-chatbot`
- Endpoint behavior: OpenAI-compatible chat completion

Example payload shape:

```json
{
  "model": "physics-chatbot",
  "messages": [
    {"role": "user", "content": "Explain Gauss's law."}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

## 5. Suggested Workflow

Use a simple chain:

```text
Text Prompt -> OpenAI Compatible Chat Node -> Preview Text
```

## 6. Import The Sample Workflow

Import:

```text
comfyui/physics_chatbot_workflow.json
```

That file includes placeholder node types and comments because custom node IDs and exact widget schemas differ across ComfyUI extensions.

## 7. What To Replace In The Placeholder Workflow

Replace:

- `TextPromptNode_PLACEHOLDER` with your text input node
- `OpenAIChatCompletionNode_PLACEHOLDER` with your OpenAI-compatible API node
- `PreviewTextNode_PLACEHOLDER` with any node that displays text output

Keep:

- Base URL: `http://127.0.0.1:8000/v1`
- Model: `physics-chatbot`

## 8. Common Issues

### Connection refused

- Make sure `scripts/06_serve_openai_api.py` is still running
- Check that the node is using `127.0.0.1` and port `8000`

### Node expects a full API key flow

- Some custom nodes insist on an API key field even for local use
- If the node allows it, leave the key blank or use a dummy local string such as `local-dev`

### Wrong response format

- Use a node that supports OpenAI `chat/completions`
- Do not point a text-generation-only node at `/v1/chat/completions`
