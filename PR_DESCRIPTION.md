# Add OpenApp Environment to OpenEnv

## Overview

This PR adds the **OpenApp Environment** to OpenEnv, integrating the [OpenApps](https://github.com/facebookresearch/OpenApps) framework for UI agent training with web applications.

**OpenApps Resources:**
- **Paper**: [OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability](https://arxiv.org/abs/2511.20766)
- **GitHub**: [facebookresearch/OpenApps](https://github.com/facebookresearch/OpenApps)
- **Demo**: [OpenApps Demo Page](https://facebookresearch.github.io/OpenApps/)

## What is OpenApp Environment?

OpenApp Environment provides a simulated web application ecosystem where agents can interact with various apps (calendar, todo, messenger, maps) using browser-based actions. It wraps the OpenApps framework and BrowserGym to create a standardized OpenEnv-compatible environment for:

- Training and evaluating UI agents
- Testing web automation strategies
- Researching human-computer interaction
- Developing multimodal agents

## Key Features

- **Multiple Web Apps**: Calendar, todo list, messenger, and map applications
- **Browser-Based Actions**: Click, fill forms, navigate, scroll, and more
- **Task-Based Evaluation**: Optional task goals with automatic reward calculation
- **Docker Support**: Fully self-contained Docker image with both OpenApps and environment server
- **BrowserGym Integration**: Built on top of BrowserGym for robust browser interaction
- **HTTP Client Interface**: Compatible with OpenEnv's standard client API
- **Web Interface**: Interactive UI for manual testing and visualization

## Changes Made

### New Files Added

```
envs/openapp_env/
├── __init__.py                   # Package exports
├── client.py                     # HTTP client for connecting to OpenApp
├── models.py                     # Data models for actions and observations
├── pyproject.toml                # Package dependencies and configuration
├── openenv.yaml                  # OpenEnv environment configuration
├── test_openapp_env.py           # Unit tests
├── README.md                     # Documentation
├── IMPLEMENTATION.md             # Implementation details and design decisions
├── assets/                       # Images and media
│   ├── OpenApps_OpenEnv_RL.png
│   └── openapps-demo.gif
└── server/                       # Server-side implementation
    ├── __init__.py
    ├── app.py                    # FastAPI server application
    ├── openapp_environment.py    # Core environment logic
    ├── Dockerfile                # Docker image definition
    └── start.sh                  # Container startup script
```

### Updated Files

- **`.github/workflows/docker-build.yml`**: Added openapp-env to CI/CD build matrix
- **`docs/environments.md`**: Added OpenApp environment card to documentation
- **`examples/openapp_example.py`**: Example demonstrating both Docker and local modes
- **`examples/openapp_recording_demo.py`**: Demo for recording videos of agent interactions

## Architecture

The OpenApp environment uses a dual-server architecture:

1. **OpenApps Server** (port 5001): Provides the web applications (calendar, todo, messenger, maps)
2. **FastAPI Server** (port 8000): Exposes the OpenEnv HTTP API

In Docker mode, both servers run inside the container automatically. In local mode, users must start the OpenApps server separately.

## Usage Examples

### Docker Mode (Recommended)

```python
from openapp_env import OpenAppAction, OpenAppEnv

# Create environment from Docker image
env = OpenAppEnv.from_docker_image("openapp-env:latest")

# Reset to initial state
result = env.reset()

# Navigate to calendar app
result = env.step(OpenAppAction(
    action_type="goto",
    url="http://localhost:5001/calendar"
))

# Cleanup
env.close()
```

### Building the Docker Image

```bash
docker build -t openapp-env:latest -f envs/openapp_env/server/Dockerfile .
```

### Running the Example

```bash
# Docker mode (recommended)
python examples/openapp_example.py --mode docker --num-steps 20

# Local mode (requires OpenApps server running separately)
export OPENAPPS_URL=http://localhost:5001
python examples/openapp_example.py --mode local
```

## Action Types Supported

- **click**: Click on an element (requires `bid`)
- **fill**: Fill a text input field (requires `bid`, `text`)
- **select_option**: Select from dropdown (requires `bid`, `value`)
- **goto**: Navigate to a URL (requires `url`)
- **scroll**: Scroll the page (requires `direction`)
- **send_keys**: Send keyboard input (requires `text`)
- **noop**: No operation

## Observations

Each observation includes:

- **html**: Current page HTML content
- **url**: Current page URL
- **open_pages_urls**: List of all open page URLs
- **active_page_index**: Index of currently active page
- **screenshot**: Base64-encoded screenshot (optional)
- **axtree_txt**: Accessibility tree for element interaction
- **app_state**: Current state of all apps (events, todos, messages, etc.)
- **task_info**: Information about current task (if using tasks)
- **last_action_error**: Error message if last action failed

## Dependencies

The environment requires:

- **Core**: `openenv-core>=0.1.1,<0.2.0` (pinned due to openai dependency conflict)
- **Web Framework**: FastAPI, Uvicorn, Pydantic
- **Browser Automation**: BrowserGym, Playwright
- **OpenApps**: Installed from GitHub (includes AgentLab dependency)

Note: Using `openenv-core>=0.1.1,<0.2.0` to avoid openai version conflict. OpenApps requires `openai<2` via agentlab, while `openenv-core==0.2.0` requires `openai>=2.7.2`.

## Testing

```bash
# Install the environment
cd envs/openapp_env
pip install -e .

# Run unit tests
python test_openapp_env.py

# Run example
python examples/openapp_example.py --mode docker
```

## Docker Build Details

- **Base image**: `python:3.11-slim`
- **Size**: ~5.7GB (includes Chromium browser and dependencies)
- **Ports**: 8000 (FastAPI), 5001 (OpenApps)
- **Multi-platform**: Supports linux/amd64 and linux/arm64
- **Health check**: Automatic readiness checks on port 8000

## CI/CD Integration

The environment is integrated into the GitHub Actions workflow:

- Automatically builds on pushes to main
- Published to GitHub Container Registry as `ghcr.io/openenv/openenv-openapp-env:latest`
- Uses cached layers for faster builds
- Supports multi-platform builds (amd64/arm64)

## Implementation Notes

### Dual Import Pattern

The code supports both in-repo development and standalone/Docker deployment using a try/except import pattern:

```python
try:
    from core.client_types import StepResult  # In-repo mode
    from .models import OpenAppAction  # Relative imports
except ImportError:
    from openenv_core.client_types import StepResult  # Standalone mode
    from openapp_env.models import OpenAppAction  # Absolute imports
```

This ensures the environment works both during development and when installed as a package.

### Docker Context

The Dockerfile uses the project root (`.`) as build context and copies the environment directory:

```dockerfile
COPY envs/openapp_env/ .
COPY envs/openapp_env/server/start.sh /app/start.sh
```

This allows the environment to access the openenv-core package during build.

## Related Work

This environment builds upon:

- **OpenApps** ([GitHub](https://github.com/facebookresearch/OpenApps)): Web application simulation framework
- **BrowserGym** ([GitHub](https://github.com/ServiceNow/BrowserGym)): Browser automation environment
- **AgentLab** ([GitHub](https://github.com/ServiceNow/AgentLab)): Agent evaluation framework

## Future Enhancements

Potential improvements for future PRs:

- [ ] Add support for custom task definitions
- [ ] Implement VNC visualization for Docker mode
- [ ] Add more example agents and evaluation scripts
- [ ] Support for additional OpenApps configurations
- [ ] Integration with OpenEnv benchmarking tools
- [ ] Upgrade to openenv-core 0.2.0 when OpenApps updates openai dependency

## Citation

If you use this environment in your research, please cite both OpenEnv and OpenApps:

```bibtex
@article{ullrich2025openapps0,
  title   = {OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability},
  author  = {Karen Ullrich and Jingtong Su and Claudia Shi and Arjun Subramonian and Amir Bar and Ivan Evtimov and Nikolaos Tsilivis and Randall Balestriero and Julia Kempe and Mark Ibrahim},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2511.20766}
}
```

## Checklist

- [x] Added new environment in `envs/openapp_env/`
- [x] Created Dockerfile with full setup
- [x] Added startup script for Docker container
- [x] Implemented HTTP client interface
- [x] Added example scripts demonstrating usage
- [x] Updated CI/CD workflow to build Docker image
- [x] Added documentation in README.md and IMPLEMENTATION.md
- [x] Updated `docs/environments.md` with environment card
- [x] Tested Docker build and execution
- [x] Tested local mode execution
- [x] Added unit tests

## Screenshots

### OpenApps Demo
![OpenApps Demo](envs/openapp_env/assets/openapps-demo.gif)

### Environment Architecture
![OpenApps OpenEnv RL](envs/openapp_env/assets/OpenApps_OpenEnv_RL.png)

---
