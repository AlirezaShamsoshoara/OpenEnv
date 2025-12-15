# OpenApp Environment for OpenEnv: UI Agent Training with Reinforcement Learning

## Project Overview

Integration of **OpenApps** (FAIR's web application simulator) into **OpenEnv** for training UI agents with reinforcement learning.

**Resources**:
- **GitHub**: [AlirezaShamsoshoara/OpenEnv (branch: alidev_openapp_env_01)](https://github.com/AlirezaShamsoshoara/OpenEnv/tree/alidev_openapp_env_01/src/envs/openapp_env)
- **Docker Image**: `vmvm-registry.fbinfra.net/hackathon/pe_fair/alisol/openapp-env:latest`
- **OpenApps**: [GitHub](https://github.com/facebookresearch/OpenApps) | [Paper](https://arxiv.org/pdf/2511.20766) | [Project Page](https://facebookresearch.github.io/OpenApps/)

## Why This Matters

**Problem**: LLM-based agents struggle with UI reliability in real-world web interactions.

**Solution**: OpenApp provides realistic web applications (calendar, todo, messenger, maps) with environment variations to train and evaluate agents through RL.

**Key Innovation**: Bridges FAIR research (OpenApps, BrowserGym) with production infrastructure (OpenEnv, Docker) - making UI agent training scalable and reproducible.

## Technical Highlights

### Production-Ready API

```python
from envs.openapp_env import OpenAppAction, OpenAppEnv

# Deploy with Docker - fully isolated
client = OpenAppEnv.from_docker_image(
    "openapp-env:latest"
)

# Standard RL interface
obs = client.reset()
obs = client.step(OpenAppAction(action_type="click", bid="calendar-btn"))
print(obs.reward)  # Task-based rewards
```

### Key Features

**Actions**: click, fill, select_option, goto, scroll, send_keys, noop

**Observations**: HTML, accessibility tree, screenshots, app state, task info

**Integration**: BrowserGym, Playwright/Chromium, compatible with TRL, torchforge, SkyRL

**Deployment**: Dual-server Docker container (OpenApps + FastAPI), auto-start, 5.7GB image

### Example: RL Training

```python
env = OpenAppEnv.from_docker_image(
    "openapp-env:latest"
)

for episode in range(1000):
    obs = env.reset()
    while not obs.done:
        action = agent.select_action(obs.observation.axtree_txt,
                                     obs.observation.app_state)
        obs = env.step(action)
        agent.update(obs.reward, obs.done)

env.close()
```

## Use Cases

1. **Task-Oriented Agents**: "Add meeting with Dennis on Friday at 2pm"
2. **Multimodal Learning**: Vision-language models for UI understanding
3. **Hierarchical RL**: Planning → navigation → execution
4. **Generalization**: Test across varied UI patterns and multi-step tasks

## Impact & Value

**Impact**: Addresses critical gap in making LLM agents reliable for web interaction at scale

**Completeness**: Fully functional with type-safe models, dual-server architecture, comprehensive docs, example code

**Community**: Strengthens OpenEnv ecosystem, compatible with major RL frameworks, enables future UI automation research

**Extensibility**: Add custom web apps, tasks, reward functions, observation modalities

## Quick Start

```bash
# Clone and run
git clone -b alidev_openapp_env_01 https://github.com/AlirezaShamsoshoara/OpenEnv.git
cd OpenEnv
docker pull vmvm-registry.fbinfra.net/hackathon/pe_fair/alisol/openapp-env:latest
python examples/openapp_example.py --mode docker
```
---

**Bottom Line**: OpenApp bridges cutting-edge research with production tooling, enabling scalable UI agent training for reliable real-world web interaction.
