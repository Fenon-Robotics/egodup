# Contributing to egodup

Thanks for helping improve egodup. The project welcomes focused bug fixes, tests, documentation, evaluation fixtures with clear redistribution rights, and carefully measured algorithm improvements.

## Before opening a pull request

1. Open an issue for substantial behavior or architecture changes.
2. Keep perceptual evidence separate from fraud, authorship, or intent claims.
3. Never commit private/vendor video, extracted evidence frames, model weights, credentials, or generated databases.
4. Add tests for changed behavior and document numerical or compatibility changes.
5. Preserve upstream attribution when modifying vendored/adapted components.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src tests tools
pytest
python -m build
```

Tests that require the SSCD checkpoint should use the explicit model setup flow; ordinary unit tests must not trigger network access.

## Pull requests

- Keep each pull request scoped to one coherent change.
- Explain the user-visible behavior and evidence contract.
- Include reproducible commands and measured results where performance or recall is discussed.
- Label synthetic fixtures as engineering tests rather than deployment validation.
- Update `THIRD_PARTY_NOTICES.md` and `third_party/UPSTREAM.md` when upstream code is introduced or changed.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
