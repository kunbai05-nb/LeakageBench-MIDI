#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text())
    failures = result.pop("parse_failures")
    exact = Counter(item["error"] for item in failures)
    classes = Counter(error.split(":", 1)[0] for error in exact.elements())
    audit = {
        "files": result["files"],
        "valid_files": result["valid_files"],
        "failed_files": len(failures),
        "failure_rate": len(failures) / result["files"],
        "exception_classes": dict(classes.most_common()),
        "top_exact_errors": dict(exact.most_common(20)),
    }
    result["parse_failure_count"] = len(failures)
    result["parse_failure_summary"] = audit
    directory = args.result_json.parent
    (directory / "RESULTS_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (directory / "PARSE_FAILURE_SUMMARY.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
