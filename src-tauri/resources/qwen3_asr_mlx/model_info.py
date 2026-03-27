#!/usr/bin/env python3
import json
import os
import sys

from huggingface_hub import HfApi


def main() -> int:
    if len(sys.argv) <= 1 or not sys.argv[1].strip():
        print("usage: model_info.py <repo_id> [endpoint]", file=sys.stderr, flush=True)
        return 2
    repo_id = sys.argv[1].strip()
    endpoint = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    ).rstrip("/")

    api = HfApi(endpoint=endpoint)
    info = api.model_info(repo_id, files_metadata=True)
    revision = info.sha
    siblings = info.siblings

    files = []
    total = 0
    for sibling in siblings:
        filename = sibling.rfilename
        size = int(sibling.size)
        files.append({"filename": filename, "size": size})
        total += size

    print(
        json.dumps(
            {
                "revision": revision,
                "files": files,
                "total": total,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
