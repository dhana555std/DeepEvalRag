"""Amazon Bedrock judge.

Uses the Bedrock *Converse* API through ``boto3``, which is model-agnostic: any Bedrock text model
(Anthropic Claude, Amazon Nova, Meta Llama, Mistral, ...) works by changing ``model`` in
``config/judges.yaml`` / ``BEDROCK_MODEL`` - no code change.

Authentication (no ``api_key_env``): the standard AWS credential chain, i.e. any of
``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY`` (+ ``AWS_SESSION_TOKEN``), ``AWS_PROFILE`` /
SSO, a Bedrock API key in ``AWS_BEARER_TOKEN_BEDROCK``, or an IAM role (EC2/ECS/EKS, or GitHub
OIDC in CI). The region comes from ``extra.region`` (env ``AWS_REGION``, default ``us-east-1``).

Notes
-----
* ``model`` is a Bedrock model id or an inference-profile id (e.g. ``us.anthropic.claude-...``);
  the model must be enabled for your account in that region.
* boto3 is synchronous, so the async path runs the call in a worker thread.
* Bedrock has no JSON mode, so the prompt is suffixed with a reminder to answer with JSON only;
  DeepEval's parser tolerates surrounding prose. Only ``text`` content blocks are used (reasoning
  blocks, if a model returns them, are ignored).
* Throttling/transient errors are retried by the base class with exponential backoff.
"""

from __future__ import annotations

import asyncio
from typing import Any

import boto3
from botocore.config import Config

from ragqa.judges.base import BaseJudge, JudgeConfigError, judges

_JSON_HINT = "\n\nRespond with a single valid JSON object only, no extra text."


@judges.register("bedrock")
class BedrockJudge(BaseJudge):
    provider = "bedrock"
    requires_api_key = False  # credentials come from the AWS chain, validated in _build_client

    def _build_client(self) -> Any:
        region = self.cfg.extra.get("region") or None
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            raise JudgeConfigError(
                "bedrock judge found no AWS credentials: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                "AWS_PROFILE, AWS_BEARER_TOKEN_BEDROCK, or run with an IAM role"
            )
        if not session.region_name:
            raise JudgeConfigError("bedrock judge needs a region: set AWS_REGION (or extra.region)")
        return session.client(
            "bedrock-runtime",
            config=Config(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=120),
        )

    def _request(self, prompt: str) -> dict[str, Any]:
        return {
            "modelId": self.cfg.model,
            "messages": [{"role": "user", "content": [{"text": prompt + _JSON_HINT}]}],
            "inferenceConfig": {"maxTokens": self.cfg.max_tokens, "temperature": self.cfg.temperature},
        }

    @staticmethod
    def _text(res: dict[str, Any]) -> str:
        blocks = res.get("output", {}).get("message", {}).get("content", [])
        return "".join(b["text"] for b in blocks if "text" in b)

    def _complete(self, prompt: str) -> str:
        return self._text(self._client.converse(**self._request(prompt)))

    async def _acomplete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete, prompt)
