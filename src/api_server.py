from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import generate_chat_response, load_chat_model


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "physics-chatbot"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = Field(default=512, alias="max_tokens")
    stream: bool = False


class ModelRunner:
    def __init__(
        self,
        base_model: str,
        served_model_name: str,
        system_prompt: str,
        adapter_path: str | None = None,
        merged_model_path: str | None = None,
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        logger=None,
    ) -> None:
        self.base_model = base_model
        self.served_model_name = served_model_name
        self.system_prompt = system_prompt
        self.logger = logger
        self.model, self.tokenizer, self.runtime = load_chat_model(
            base_model_name=base_model,
            adapter_path=adapter_path,
            merged_model_path=merged_model_path,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            logger=logger,
        )

    def chat(self, messages: list[dict[str, Any]], temperature: float, top_p: float, max_tokens: int) -> dict[str, Any]:
        return generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name=self.base_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            system_prompt=self.system_prompt,
        )


def create_app(
    base_model: str,
    served_model_name: str,
    system_prompt: str,
    adapter_path: str | None = None,
    merged_model_path: str | None = None,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    logger=None,
) -> FastAPI:
    runner = ModelRunner(
        base_model=base_model,
        served_model_name=served_model_name,
        system_prompt=system_prompt,
        adapter_path=adapter_path,
        merged_model_path=merged_model_path,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        logger=logger,
    )

    app = FastAPI(title="PhysicsGPT Local API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": runner.served_model_name,
            "base_model": runner.base_model,
            "runtime": runner.runtime,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": runner.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not implemented in this local server yet.")
        if not request.messages:
            raise HTTPException(status_code=400, detail="At least one chat message is required.")

        normalized_messages = [
            {
                "role": message.role,
                "content": message.content if isinstance(message.content, str) else str(message.content),
            }
            for message in request.messages
        ]
        result = runner.chat(
            messages=normalized_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runner.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        }

    return app
