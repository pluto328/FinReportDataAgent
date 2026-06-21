"""Deprecated: use eval/retrieval_weight_eval.py for weight tuning experiments."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "rag_metric_eval.py 已弃用。\n"
        "请使用:\n"
        "  poetry run python eval/gen_testset.py\n"
        "  poetry run python eval/retrieval_weight_eval.py",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
