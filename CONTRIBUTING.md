# Contributing

Use Python 3.10 or newer. Install with `python -m pip install -e ".[dev]"`, create focused tests for behavior changes, and run `pytest` before proposing a change.

Scientific calculations must not be changed solely to satisfy a test. Document changes to formulas, defaults, random seeds, configuration keys, or output columns in `CHANGELOG.md` and `docs/migration.md`.

Do not commit research datasets, generated outputs, credentials, virtual environments, caches, or large binary artifacts.
