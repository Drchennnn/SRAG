"""Extend 7da7d05's observer for public fixtures with the outgoing JSON and parser stage."""

import runpy
import sys
from pathlib import Path

EXTRA_OBSERVERS = """
from hashlib import sha256

def is_generation(messages):
    return bool(messages and messages[0]["content"].startswith(workspace_generation._INSTRUCTIONS))

original_transport = httpx.AsyncHTTPTransport.handle_async_request
async def observed_transport(self, request):
    if request.method == "POST":
        try:
            payload = loads(request.content)
        except (TypeError, ValueError):
            payload = {}
        if isinstance(payload, dict) and is_generation(payload.get("messages")):
            write_stage("outgoing_generation_payload", {
                "body_sha256": sha256(request.content).hexdigest(), "payload": payload,
            })
    return await original_transport(self, request)
httpx.AsyncHTTPTransport.handle_async_request = observed_transport

original_completion = DeepSeekClient._chat_completion
async def observed_completion(self, *args, **kwargs):
    result = await original_completion(self, *args, **kwargs)
    if is_generation(kwargs.get("messages")):
        write_stage("provider_content_before_client_normalization", result)
    return result
DeepSeekClient._chat_completion = observed_completion

original_parse = workspace_generation._parse_answer
def observed_parse(content, allowed, **kwargs):
    try:
        statements, missing, experiment_missing = original_parse(content, allowed, **kwargs)
    except (TypeError, ValueError, RecursionError) as error:
        write_stage("parsed_reply", {"error": type(error).__name__})
        raise
    write_stage("parsed_reply", {
        "statements": [vars(item) for item in statements],
        "missing": missing, "experiment_missing": experiment_missing,
    })
    return statements, missing, experiment_missing
workspace_generation._parse_answer = observed_parse
"""


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Pass the observer from 7da7d05, followed by its normal flags.")
    observer_path = Path(sys.argv.pop(1))
    run = runpy.run_path(str(observer_path))["main"]
    run.__globals__["OBSERVERS"] += EXTRA_OBSERVERS
    run()


if __name__ == "__main__":
    main()
