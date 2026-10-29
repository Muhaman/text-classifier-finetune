"""Zero-shot Claude baseline on Banking77 test examples. Not run in CI.

Needs the ``baseline`` extra and ``ANTHROPIC_API_KEY``:

    uv run --extra baseline python scripts/zero_shot_baseline.py --n 200
"""

from clf.baseline import main

if __name__ == "__main__":
    main()
