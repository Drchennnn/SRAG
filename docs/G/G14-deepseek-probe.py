"""Run the configured DeepSeek client with synthetic text and log no credentials."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx

from app.clients.deepseek import DeepSeekClient
from app.config import Settings


class ObservedTransport(httpx.AsyncHTTPTransport):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = perf_counter()
        entry = {"method": request.method, "path": request.url.path}
        try:
            response = await super().handle_async_request(request)
            entry["status"] = response.status_code
            return response
        except httpx.TransportError as error:
            entry["failure_type"] = type(error).__name__
            raise
        finally:
            entry["headers_ms"] = round((perf_counter() - started) * 1000, 3)
            self.attempts.append(entry)


async def main(env_file: Path) -> dict:
    settings = Settings(_env_file=env_file)
    result = {
        "deepseek_configured": bool(settings.deepseek_api_key.strip()),
        "jev_configured": bool(settings.jev_api_key.strip()),
        "query_model": settings.deepseek_query_model,
        "answer_model": settings.deepseek_chat_model,
        "timeout_seconds": settings.deepseek_timeout_seconds,
        "max_retries": settings.deepseek_max_retries,
        "calls": {},
    }
    if not result["deepseek_configured"]:
        return result
    for kind in ("query", "answer"):
        transport = ObservedTransport()
        client = DeepSeekClient(settings, transport=transport)
        probe = {"attempts": transport.attempts, "status": "failed"}
        result["calls"][kind] = probe
        try:
            if kind == "query":
                reply = await client.understand_query(
                    "合成校验样本的电容值是多少？", enable_rewrite=True
                )
                if "error" not in reply and isinstance(reply.get("key_concepts"), list):
                    probe["status"] = "succeeded"
            else:
                question = "用中文回答：合成记录的电容是 47 nF，电容是多少？"
                reply = await client.generate_answer([{"role": "user", "content": question}])
                if "47" in reply["content"] and "nF" in reply["content"]:
                    probe["status"] = "succeeded"
            if isinstance(reply.get("usage"), dict):
                probe["total_tokens"] = reply["usage"].get("total_tokens")
        except (RuntimeError, ValueError, httpx.HTTPError) as error:
            probe["failure_type"] = type(error).__name__
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(main(args.env_file)), ensure_ascii=False, indent=2))
