---
description: Run the full verification loop (format, lint, types, tests) and report
---

Run the project's verification loop from the repo root and report concisely what
passed or failed. Run all four even if an earlier one fails:

1. `uv run ruff format --check .`
2. `uv run ruff check .`
3. `uv run pyright`
4. `uv run pytest`

If everything passes, say so in one line. If anything fails, summarize the
failures (file:line + rule/message) and offer to fix them — don't fix without
asking unless the fix is purely mechanical (formatting / autofixable lint).
