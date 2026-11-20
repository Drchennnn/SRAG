"""Exercise real Jev scoring and a hot repeat using only synthetic passages."""

import argparse
import json
from pathlib import Path

from app.clients.jev import JevCandidate, JevClient
from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    settings = Settings(_env_file=parser.parse_args().env_file)
    client = JevClient(settings)
    question = "公开合成记录中，Oriole 环的电压上限是多少？"
    pool = [
        JevCandidate(
            "candidate_fact", "Oriole ring limits", "The Oriole ring voltage limit is 47 V."
        ),
        JevCandidate("candidate_decoy", "Oriole storage", "Store the Oriole ring in a blue box."),
    ]
    first = client.score(question, pool)
    again = client.score(question, pool)
    print(
        json.dumps({"cold": first.trace, "repeat": again.trace, "scores": first.scores}, indent=2)
    )


if __name__ == "__main__":
    main()
