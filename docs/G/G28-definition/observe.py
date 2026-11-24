"""Capture generation boundaries for a public fixture replay, with real browser and model calls."""

import argparse
import json
import runpy
import sys
from pathlib import Path

OBSERVERS = """
from app.clients.deepseek import DeepSeekClient
from app.rag import workspace_generation
from json import dumps, loads

def write_stage(stage, value):
    with (out / "stages.jsonl").open("a", encoding="utf-8") as event:
        event.write(dumps({"stage": stage, "value": value}, ensure_ascii=False) + chr(10))

original_answer = DeepSeekClient.generate_answer
async def observed_answer(self, messages, **kwargs):
    write_stage("model_input", {"messages": messages, "options": kwargs})
    reply = await original_answer(self, messages, **kwargs)
    write_stage("provider_reply", reply.get("content"))
    return reply
DeepSeekClient.generate_answer = observed_answer

original_filter = workspace_generation._supported_technical_statements
def observed_filter(statements, evidence, question):
    result = original_filter(statements, evidence, question)
    write_stage("technical_filter", {
        "before": [vars(item) for item in statements],
        "after": [vars(item) for item in result],
    })
    return result
workspace_generation._supported_technical_statements = observed_filter
"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    runner = runpy.run_path(str(args.runner.resolve()))
    server_script = runner["server_script"]
    replay = runner["replay"]
    run = runner["main"]

    def server_with_observers(*params):
        source = server_script(*params)
        index = source.rindex("uvicorn.run(")
        return source[:index] + OBSERVERS + source[index:]

    def replay_with_stages(url, session, folder, manifest, log, record, save):
        stage_log = log.with_name("stages.jsonl")

        def capture():
            if stage_log.exists():
                record["generation_stages"] = [
                    json.loads(line) for line in stage_log.read_text().splitlines()
                ]
            save()

        replay(url, session, folder, manifest, log, record, capture)

    sys.argv = [sys.argv[0], *forwarded]
    run.__globals__.update(server_script=server_with_observers, replay=replay_with_stages)
    run()


if __name__ == "__main__":
    main()
