# Contributing

## Workflow

- `main` is the only long-lived branch. One pull request per phase (F0…F10);
  small fixes go in their own PR. Branches are deleted after merge.
- Every PR that closes a phase includes `reports/fN_*.json` with the metrics of
  that phase's Definition of Done and a "DoD: met / not met" section in the
  description. Numbers come from scripts in `scripts/`, never typed by hand.
- Changes to thresholds, stack or protocol go through an ADR in `docs/adr/`,
  and never after seeing evaluation results.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`).

## Local checks

```bash
uv sync
uv run pre-commit install
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Tests that need the simulation image are marked `@pytest.mark.sim` and are
excluded from the `verify` CI job; they run inside the image
(`docker run … uv run pytest -m sim`).

## Rules of evidence

- No fabricated numbers. A metric without a report file and a producing script
  does not exist.
- Separate what was measured from what is estimated or still to verify.
- Never commit secrets; use `.env` (ignored) and keep `.env.example` current.
