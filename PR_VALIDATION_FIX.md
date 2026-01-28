# Fix Python 3.10 compatibility for tomllib import

## Summary

Fixes a `ModuleNotFoundError` when running OpenEnv CLI commands on Python 3.10 by adding a fallback import for the `tomli` package.

## Problem

The `tomllib` module is only available in Python 3.11+. When running on Python 3.10, the following error occurs:

```
$ openenv push
Traceback (most recent call last):
  ...
  File "/src/openenv/cli/_validation.py", line 15, in <module>
    import tomllib
ModuleNotFoundError: No module named 'tomllib'
```

## Solution

Add a try/except import that falls back to `tomli` (the third-party backport) when `tomllib` is not available:

```python
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
```

## Changes

- `src/openenv/cli/_validation.py`: Updated import to support Python 3.10

## Test Plan

- [x] Tested `openenv push` on Python 3.10 - works correctly
- [x] Tested `openenv push` on Python 3.11+ - still works (uses built-in `tomllib`)
- [x] `tomli` is already listed as a dependency in `pyproject.toml`

## Notes

- The `tomli` package is already a dependency of the project, so no new dependencies are required
- This maintains backward compatibility with Python 3.10 which is listed as the minimum supported version (`requires-python = ">=3.10"`)
