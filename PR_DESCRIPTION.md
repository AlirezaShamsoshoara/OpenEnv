# PR: Add Screenshot Support to OpenApp Environment

## Summary

This PR adds screenshot capture functionality to the OpenApp environment and fixes dependency issues that were causing import errors.

### Key Changes

- **Screenshot Feature**: Environment now captures and returns base64-encoded PNG screenshots after each action (`reset()` and `step()`)
- **Dependency Fixes**: Resolved `beartype` version conflicts and added missing `fastmcp` dependency to Dockerfile
- **Documentation**: Added troubleshooting guide and screenshot usage instructions to README

## Changes

### Screenshot Implementation (`envs/openapp_env/server/openapp_environment.py`)

- Added `_current_screenshot` instance variable to track screenshot state
- Added `_extract_screenshot()` helper method to convert BrowserGym numpy arrays to base64 PNG
- Updated `reset()` and `step()` to extract and return screenshots
- Updated all `_execute_*` methods (click, fill, goto, scroll, send_keys) to capture screenshots
- Updated `_update_observation_from_page()` to capture screenshots from Playwright

### Dockerfile Fixes (`envs/openapp_env/server/Dockerfile`)

- Added `fastmcp` dependency (required by openenv-core for MCP support)
- Added beartype force-reinstall step after all pip installs to fix version conflicts
- beartype is pinned to `>=0.15,<0.18` for py-key-value-aio compatibility

### Dependencies (`envs/openapp_env/pyproject.toml`)

- Added `Pillow>=10.0.0` for screenshot image processing

### Example Script (`examples/openapp_example.py`)

- Added `--test-screenshots` flag to verify screenshot functionality
- Screenshots saved to `examples/screenshot_output/` with descriptive names
- Works with both local and docker modes

### Documentation (`envs/openapp_env/README.md`)

- Added "Screenshots" section with usage examples and test instructions
- Added troubleshooting section for `beartype_this_package` import errors
- Added `PYTHONNOUSERSITE` fix for conda/virtualenv users

### Other

- Added beartype pin to main `pyproject.toml` for local development

## Test Plan

### Local Mode
```bash
# Terminal 1: Start OpenApps server
cd /path/to/OpenApps
uv run launch.py

# Terminal 2: Test screenshots
export OPENAPPS_URL=http://localhost:5001
export PYTHONNOUSERSITE=1
python examples/openapp_example.py --mode local --test-screenshots
```

### Docker Mode
```bash
# Build the updated Docker image
cd envs/openapp_env
docker build -t openapp-env:latest -f server/Dockerfile .

# Test screenshots
export PYTHONNOUSERSITE=1
python examples/openapp_example.py --mode docker --test-screenshots
```

### Expected Output
- Screenshots saved to `examples/screenshot_output/`
- Files like `local_reset.png`, `local_step_1_goto.png`, etc.
- Each screenshot should be a valid PNG image of the browser state

## Breaking Changes

None. The screenshot field was already defined in `OpenAppObservation` but was not populated. This PR now populates it.

## Related Issues

- Partner reported missing screenshots when using Docker mode
- `beartype_this_package` import error when running with certain package configurations
