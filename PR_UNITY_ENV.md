# Add Unity ML-Agents Environment

## Summary

This PR adds a new Unity ML-Agents environment wrapper to OpenEnv, providing access to Unity's reinforcement learning environments (PushBlock, 3DBall, GridWorld, etc.) through the standardized OpenEnv HTTP/WebSocket interface.

## Features

- **Full Unity ML-Agents Integration**: Wraps all environments from the ML-Agents default registry (17+ environments)
- **Multiple Deployment Modes**:
  - **Direct Mode**: Run Unity environments directly in-process (recommended for local development)
  - **Server Mode**: Client-server architecture via HTTP/WebSocket
  - **Docker Mode**: Containerized deployment for production/cloud environments
- **Action Space Support**: Both discrete (PushBlock, GridWorld) and continuous (3DBall) action spaces
- **Dynamic Environment Switching**: Switch between environments at runtime without restarting
- **Headless Mode**: Run without graphics for faster training
- **HuggingFace Spaces Ready**: Configured for deployment on HuggingFace Spaces

## New Files

```
envs/unity_env/
├── README.md              # Comprehensive documentation
├── pyproject.toml         # Package configuration
├── client.py              # EnvClient implementation
├── models.py              # Action, Observation, State models
├── assets/                # Demo GIFs
│   ├── unity_pushblock.gif
│   └── unity_3dball.gif
└── server/
    ├── Dockerfile         # Docker configuration
    ├── app.py             # FastAPI server
    └── unity_environment.py  # Core environment wrapper

examples/
└── unity_simple.py        # Example usage script

tests/envs/
└── test_unity_environment.py  # Comprehensive test suite (19 tests)
```

## Supported Environments

| Environment | Action Type | Description |
|------------|-------------|-------------|
| PushBlock | Discrete (7) | Push a block to a goal position |
| 3DBall | Continuous (2) | Balance a ball on a platform |
| 3DBallHard | Continuous (2) | Harder version of 3DBall |
| GridWorld | Discrete (5) | Navigate a grid to find goals |
| Basic | Discrete (3) | Simple left/right movement |
| + 12 more | Various | All ML-Agents registry environments |

## Usage Examples

```python
# Direct mode (simplest)
from envs.unity_env.client import UnityEnv
from envs.unity_env.models import UnityAction

env = UnityEnv.from_direct(no_graphics=True)
result = env.reset(env_id="PushBlock")
action = UnityAction(discrete_actions=[1])  # Move forward
result = env.step(action)
env.close()

# Server mode
with UnityEnv(base_url="http://localhost:8000") as env:
    result = env.reset(env_id="3DBall")
    action = UnityAction(continuous_actions=[0.5, -0.3])
    result = env.step(action)
```

## Known Limitations

- **Apple Silicon + Docker**: Docker mode does not work on M1/M2/M3/M4 Macs due to Unity's Mono runtime crashing under x86_64 emulation. Use direct mode or server mode instead (documented in README).
- **First Run**: Downloads ~500MB of Unity binaries on first use (cached for subsequent runs)
- **Single Worker**: Unity environments are not thread-safe; use `workers=1`

## Test Plan

- [x] All 19 unit tests pass (`pytest tests/envs/test_unity_environment.py -v`)
- [x] Direct mode tested locally on macOS
- [x] Server mode tested locally
- [x] Docker mode tested on x86_64 Linux (GitHub Actions / cloud VM)
- [x] HuggingFace Spaces deployment tested
- [x] Documentation reviewed

## Dependencies

- `mlagents-envs` (installed from Unity ML-Agents git repository)
- `openenv-core[core]` (installed from git for latest features)
- `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pillow`
