# OpenEnv Presentation Slides

## Slide 1: Title

### OpenEnv: Agentic Execution Environments

**An end-to-end framework for creating, deploying, and using isolated execution environments for agentic RL training**

- Built on Gymnasium-style simple APIs
- Open source (BSD 3-Clause License)
- Backed by Meta-PyTorch, Hugging Face, and many more

---

## Slide 2: The Problem

### Why Do We Need OpenEnv?

**Challenges in Agentic RL Training:**

1. **Fragmented Interfaces** - Every environment has different APIs
2. **Training-Production Gap** - Training environments don't match production
3. **Security & Isolation** - Running untrusted agent code safely
4. **Reproducibility** - Hard to replicate results across machines

**OpenEnv Solution:**
- Standardized Gymnasium-style API (`reset`, `step`, `state`)
- Container-based isolation with Docker
- Same interface from training to production

---

## Slide 3: Architecture

### How OpenEnv Works

```
┌─────────────────────────────────────────┐
│           Client Application            │
│  ┌──────────────┐  ┌──────────────┐     │
│  │  EchoEnv     │  │  CodingEnv   │     │
│  │  (EnvClient) │  │  (EnvClient) │     │
│  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────┘
          │ WebSocket       │ WebSocket
          │                 │
┌─────────▼─────────────────▼─────────────┐
│       Docker Containers (Isolated)      │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ FastAPI      │  │ FastAPI      │     │
│  │ Environment  │  │ Environment  │     │
│  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────┘
```

**Key Components:**
- **EnvClient** - Client-side, handles WebSocket connections
- **Environment** - Server-side, implements environment logic
- **Docker Containers** - Isolated, reproducible execution

---

## Slide 4: Core Concepts

### The OpenEnv API

**Three Simple Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `reset()` | Start a new episode | Initial Observation |
| `step(action)` | Execute an action | Observation, Reward, Done |
| `state()` | Get episode metadata | State (episode_id, step_count) |

**Type-Safe Data Models:**
- `Action` - What the agent does
- `Observation` - What the agent sees
- `StepResult` - Observation + Reward + Done flag

**Design Principles:**
- Rewards computed inside the environment
- Agents cannot reset (no learning that consequences are reversible)
- One environment = one trajectory

---

## Slide 5: Quick Start

### Getting Started

**1. Connect to Environment:**
```python
from openapp_env import OpenAppAction, OpenAppEnv

# Connect to OpenApp environment
client = OpenAppEnv(base_url="http://localhost:8000")
result = client.reset()
```

**2. Interact with the App:**
```python
# Navigate to calendar app
result = client.step(OpenAppAction(action_type="goto", url="/calendar"))

# Click on a button
result = client.step(OpenAppAction(action_type="click", bid="add-event"))

# Fill in a form
result = client.step(OpenAppAction(action_type="fill", bid="title", text="Meeting"))

print(result.observation.screenshot)  # See the result
print(result.reward)  # Get reward
```

**3. Create Your Own Environment:**
```bash
openenv init my_game_env
openenv push  # Deploy to Hugging Face Spaces
```

---

## Slide 6: Ecosystem & Community

### Supported RL Frameworks

| Framework | Description |
|-----------|-------------|
| **torchforge** | PyTorch's agentic RL framework |
| **TRL** | Transformer Reinforcement Learning |
| **Unsloth** | Fast LLM fine-tuning |
| **SkyRL** | UC Berkeley RL framework |
| **ART** | OpenPipe's training platform |
| **Oumi** | RL training toolkit |

**Community & Partners:**
- Meta-PyTorch, Hugging Face, Scale AI, Patronus AI
- Lightning AI, vLLM, Stanford Scaling Intelligence Lab
- And many more...

**Get Involved:**
- GitHub: `github.com/meta-pytorch/OpenEnv`
- Discord: OpenEnv community
- Docs: `meta-pytorch.org/OpenEnv`
