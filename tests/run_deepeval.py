#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv()
    sys.path.insert(0, str(project_root()))
    from deepeval.transcript_eval import main as run_eval

    # Allow overriding models via env for reliability
    os.environ.setdefault("SUMMARIZE_MODEL", "gpt-3.5-turbo")
    os.environ.setdefault("DEEPEVAL_MODEL", "gpt-3.5-turbo")

    run_eval()


if __name__ == "__main__":
    main()


