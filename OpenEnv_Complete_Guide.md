# OpenEnv: Complete Learning Guide

## Table of Contents
1. [What is OpenEnv?](#what-is-openenv)
2. [Motivation Behind OpenEnv](#motivation-behind-openenv)
3. [Why Users Should Use OpenEnv](#why-users-should-use-openenv)
4. [Pros and Cons](#pros-and-cons)
5. [Fundamentals of OpenEnv](#fundamentals-of-openenv)
6. [Available Environments](#available-environments)
7. [How to Add a New Environment](#how-to-add-a-new-environment)
8. [Integration Examples](#integration-examples)
9. [Resources and Community](#resources-and-community)

---

## What is OpenEnv?

**OpenEnv** is an end-to-end framework for creating, deploying, and using **isolated execution environments** for agentic reinforcement learning (RL) training. It provides a standardized specification that wraps ANY existing environment (games, simulations, real-world systems) and makes them compatible with modern RL training frameworks.

### Key Innovation

OpenEnv is **NOT a game engine** or simulation platform. Instead, it's a **universal specification and framework** that:

- Provides a **unified Gymnasium-style API** (`reset()`, `step()`, `state()`)
- Enables **container-based isolation** using Docker for secure execution
- Offers an **HTTP server/client architecture** for distributed training
- Includes **70+ pre-built environments** across diverse domains
- Provides **CLI tools** for scaffolding and deployment to Hugging Face Spaces

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    Client Application                   │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  EchoEnv       │              │  CodingEnv       │   │
│  │ (HTTPEnvClient)│              │  (HTTPEnvClient) │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ HTTP                          │ HTTP
            │ (reset, step, state)          │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)               │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   EchoEnvironment    │    │ PythonCodeActEnv     │   │
│  │ (Environment base)   │    │ (Environment base)   │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Motivation Behind OpenEnv

### The Problem

Before OpenEnv, the agentic RL community faced several challenges:

1. **Fragmented Ecosystem**: Every environment had its own API, making it difficult to switch between tasks
2. **Deployment Complexity**: Setting up environments required complex installation procedures and dependency management
3. **Security Concerns**: Running untrusted agent code alongside environment logic posed security risks
4. **Reproducibility Issues**: "Works on my machine" problems plagued environment sharing
5. **Integration Overhead**: Each RL framework had to write custom adapters for each environment

### The Solution

OpenEnv addresses these challenges by:

1. **Standardizing the API**: A simple, consistent interface (`reset()`, `step()`, `state()`) across all environments
2. **Containerization**: Docker isolation ensures reproducibility and security
3. **HTTP-Based Communication**: Environments can run remotely, enabling distributed training
4. **Easy Deployment**: One-command deployment to Hugging Face Spaces
5. **Framework Agnostic**: Works with torchforge, TRL, Unsloth, SkyRL, ART, and more

### Design Philosophy

OpenEnv follows key principles:

- **Separation of Concerns**: Clear client-server boundaries
- **Type Safety**: Strongly-typed actions, observations, and state using Pydantic
- **Container Isolation**: Each environment runs in its own sandboxed container
- **Simple APIs**: Minimal, intuitive interfaces inspired by Gymnasium

---

## Why Users Should Use OpenEnv

### For RL Researchers

1. **Access to 70+ Diverse Environments**: From games (Chess, Atari) to web automation (BrowserGym) to stock trading (FinRL)
2. **Consistent API**: Learn once, use everywhere - same interface across all tasks
3. **Easy Integration**: Works with major RL frameworks (torchforge, TRL, Unsloth, SkyRL, ART)
4. **Reproducible Research**: Docker containers ensure your experiments run identically everywhere
5. **Community-Driven**: Growing collection of environments and active development

### For Environment Creators

1. **Quick Scaffolding**: `openenv init` creates a complete environment template in seconds
2. **Best Practices Built-In**: Templates follow proven patterns for robustness
3. **Automated CI/CD**: GitHub Actions automatically build and publish Docker images
4. **Easy Sharing**: One-command deployment to Hugging Face Spaces
5. **Type Safety**: Pydantic models catch errors at development time

### For Framework Developers

1. **Standard Interface**: Write integration code once, support 70+ environments
2. **Extensible**: Easy to add new environments to your framework
3. **Remote Execution**: HTTP interface enables distributed training setups
4. **Well-Documented**: Comprehensive guides and examples

---

## Pros and Cons

### Pros

#### For Users

- **Unified API**: Consistent `reset()`, `step()`, `state()` across all environments
- **Rich Ecosystem**: 70+ pre-built environments spanning multiple domains
- **Production-Ready**: Used by Meta PyTorch, Hugging Face, and major RL frameworks
- **Reproducible**: Docker ensures identical behavior across machines
- **Web Interface**: Built-in interactive UI for testing and debugging
- **Type-Safe**: Pydantic models prevent runtime errors
- **Well-Documented**: Comprehensive guides, examples, and tutorials
- **Active Community**: Supported by Meta PyTorch, growing contributor base

#### For Creators

- **Fast Development**: CLI scaffolding and templates accelerate creation
- **Minimal Boilerplate**: Core framework handles HTTP, Docker, and serialization
- **Automated Deployment**: GitHub Actions + Hugging Face integration
- **Flexible Dependencies**: Environment-specific dependency isolation
- **Testing Support**: Direct environment testing before containerization

### Cons

#### Current Limitations

1. **Early Stage**: Still experimental with potential breaking changes (as of 2025)
2. **Docker Requirement**: Must have Docker installed and running
3. **Learning Curve**: Requires understanding of Docker, FastAPI, and Pydantic
4. **Overhead**: HTTP communication adds latency vs. in-process environments
5. **Resource Usage**: Each environment runs in separate container (memory overhead)
6. **Limited Platforms**: Primarily focused on Docker and Hugging Face Spaces

#### Development Considerations

- **Complex Setup for Complex Envs**: Environments with heavy dependencies require careful Dockerfile setup
- **Debugging**: Containerized environments can be harder to debug than local code
- **Version Management**: Need to manage multiple Docker images
- **Network Dependency**: HTTP-based communication requires network stability

---

## Fundamentals of OpenEnv

### Core Concepts

#### 1. The Three Methods

Every OpenEnv environment implements three core methods:

```python
class Environment:
    def reset(self) -> Observation:
        """Initialize a new episode, return initial observation"""

    def step(self, action: Action) -> Observation:
        """Execute an action, return resulting observation"""

    @property
    def state(self) -> State:
        """Access episode metadata (episode_id, step_count, etc.)"""
```

#### 2. Type System

OpenEnv uses Pydantic dataclasses for type safety:

```python
from dataclasses import dataclass
from core.env_server.types import Action, Observation, State

@dataclass
class MyAction(Action):
    """What the agent can do"""
    command: str
    parameters: dict

@dataclass
class MyObservation(Observation):
    """What the agent sees"""
    result: str
    success: bool
    reward: float = 0.0

@dataclass
class MyState(State):
    """Episode tracking (automatically includes episode_id, step_count)"""
    custom_field: int = 0
```

#### 3. Client-Server Architecture

**Server Side (runs in Docker)**:
- FastAPI application exposing HTTP endpoints
- Environment implementation (game logic, simulation, etc.)
- Auto-generated endpoints: `/reset`, `/step`, `/state`, `/health`, `/web`

**Client Side (your training code)**:
- HTTPEnvClient subclass for type-safe communication
- Automatic Docker container management
- Serialization/deserialization of actions and observations

#### 4. HTTP Communication Flow

```
Training Script                    Docker Container
     │                                    │
     ├─── POST /reset ──────────────────> │
     │ <─── {observation: {...}} ─────────┤
     │                                    │
     ├─── POST /step {action: {...}} ───> │
     │ <─── {observation: {...},          │
     │       reward: 1.5,                 │
     │       done: false} ────────────────┤
     │                                    │
     ├─── GET /state ────────────────────> │
     │ <─── {episode_id: "...",           │
     │       step_count: 5} ──────────────┤
```

### Project Structure

```
OpenEnv/
├── src/
│   ├── core/                      # Core framework
│   │   ├── env_server/            # Server-side components
│   │   │   ├── interfaces.py      # Environment base class
│   │   │   ├── types.py           # Action, Observation, State
│   │   │   └── http_server.py     # FastAPI server creation
│   │   ├── http_env_client.py     # HTTPEnvClient base class
│   │   └── containers/            # Docker management
│   │
│   ├── envs/                      # 12 official environments
│   │   ├── echo_env/              # Simple test environment
│   │   ├── coding_env/            # Python code execution
│   │   ├── browsergym_env/        # Web automation (100+ tasks)
│   │   ├── git_env/               # Git operations
│   │   └── ...
│   │
│   └── openenv_cli/               # CLI tools
│       ├── __main__.py            # `openenv` command
│       └── templates/             # Environment templates
│
├── examples/                      # 18+ example scripts
│   ├── grpo_blackjack/            # Featured: GRPO training
│   ├── local_echo_env.py
│   └── OpenEnv_Tutorial.ipynb     # Interactive Colab
│
├── .github/workflows/             # CI/CD
│   └── docker-build.yml           # Automated image builds
│
└── pyproject.toml                 # Root dependencies
```

### Dependency Management

OpenEnv uses a **hierarchical dependency system**:

1. **Root `pyproject.toml`**: Core dependencies (FastAPI, Pydantic, Uvicorn, Docker SDK)
2. **Environment `pyproject.toml`**: Environment-specific client dependencies
3. **Server `requirements.txt`**: Container runtime dependencies

This allows:
- Clean separation of concerns
- Environment-specific dependencies without conflicts
- Flexible deployment options (local, Docker, Kubernetes)

---

## Available Environments

OpenEnv includes **12 official environments** spanning diverse domains. Here's a comprehensive breakdown:

### 1. Echo Environment

**Path**: `src/envs/echo_env/`

**Purpose**: Simple test environment that echoes messages back

**Use Cases**:
- Testing HTTP server infrastructure
- Learning the framework basics
- Verifying container deployment

**Action**:
```python
@dataclass
class EchoAction(Action):
    message: str
```

**Observation**:
```python
@dataclass
class EchoObservation(Observation):
    echoed_message: str
    message_length: int = 0
    reward: float = 0.0
```

**Reward Calculation**: `message_length * 0.1`

**Example**:
```python
from envs.echo_env import EchoAction, EchoEnv

client = EchoEnv.from_docker_image("echo-env:latest")
result = client.reset()
result = client.step(EchoAction(message="Hello!"))
print(result.observation.echoed_message)  # "Hello!"
print(result.reward)  # 0.6
client.close()
```

---

### 2. OpenSpiel Games

**Path**: `src/envs/openspiel_env/`

**Purpose**: Access to 70+ classic games via DeepMind's OpenSpiel library

**Games Include**:
- Chess
- Poker (Texas Hold'em, Leduc)
- Tic-Tac-Toe
- Blackjack
- Go
- Backgammon
- And 60+ more...

**Use Cases**:
- Game-playing agent training
- Multi-agent RL research
- Benchmarking algorithms

**Featured Example**: GRPO BlackJack training in `examples/grpo_blackjack/`

**Integration**: Used in torchforge tutorial, TRL examples

---

### 3. Connect4

**Path**: `src/envs/connect4_env/`

**Purpose**: Classic Connect 4 board game

**Use Cases**:
- Two-player game AI training
- Board game strategy learning
- Simple game environment for testing

---

### 4. Atari

**Path**: `src/envs/atari_env/`

**Purpose**: 100+ Atari 2600 games via Arcade Learning Environment (ALE)

**Games Include**:
- Pong
- Breakout
- Space Invaders
- Pac-Man
- Ms. Pac-Man
- Q*bert
- Montezuma's Revenge
- And 90+ more...

**Use Cases**:
- Classic RL benchmarking
- Visual perception training
- Pixel-based policy learning

**Configuration**: Select game via environment variables

---

### 5. BrowserGym Environment

**Path**: `src/envs/browsergym_env/`

**Purpose**: Web automation training and evaluation

**Benchmarks**:

1. **MiniWoB++** (Training - Ready to Use)
   - 100+ synthetic web tasks
   - No setup required
   - Perfect for training agents

2. **WebArena** (Evaluation - Requires Setup)
   - 812 realistic web tasks
   - 6 realistic websites (e-commerce, forums, admin panels)
   - Production-like complexity

3. **VisualWebArena**
   - 910 multimodal web tasks
   - Vision + language requirements

4. **WorkArena**
   - Enterprise software automation
   - ServiceNow-like tasks

**Actions**: Natural language action strings
- `"click [id=123]"`
- `"fill [name=search] Hello World"`
- `"scroll down"`
- `"navigate https://example.com"`

**Use Cases**:
- Training web automation agents
- Browser-based task completion
- UI/UX interaction learning

**Key Feature**: Train on MiniWoB++, evaluate on WebArena for realistic benchmarking

---

### 6. Coding Environment

**Path**: `src/envs/coding_env/`

**Purpose**: Execute arbitrary Python code in a sandboxed environment

**Action**:
```python
@dataclass
class CodeAction(Action):
    code: str  # Python code to execute
```

**Observation**:
```python
@dataclass
class CodeObservation(Observation):
    stdout: str
    stderr: str
    exit_code: int
```

**Features**:
- Safe code execution using smolagents
- Persistent execution context within episodes
- Capture stdout, stderr, and exit codes
- Error handling with detailed messages

**Use Cases**:
- Training code-generating LLMs
- Computational task solving
- Programming challenge agents

**Example**:
```python
from envs.coding_env import CodeAction, CodingEnv

client = CodingEnv.from_docker_image("coding-env:latest")
client.reset()

# Execute Python code
result = client.step(CodeAction(code="print(2 + 2)"))
print(result.observation.stdout)  # "4\n"
print(result.observation.exit_code)  # 0
```

---

### 7. Chat Environment

**Path**: `src/envs/chat_env/`

**Purpose**: Chat-based environment for LLMs with tokenization

**Features**:
- Built-in tokenization support
- Message history management
- Conversation state tracking

**Use Cases**:
- Conversation-based RL training
- Dialogue agent development
- Chat bot optimization

---

### 8. Git Environment

**Path**: `src/envs/git_env/`

**Purpose**: Git repository management via shared Gitea service

**Architecture**:
- External Gitea server (shared)
- Isolated workspace containers (per environment instance)

**Actions**:
```python
# List available repositories
list_repos()

# Clone repository to workspace
clone_repo(repo_name)

# Execute git commands
execute_git_command(command, working_dir)
```

**Use Cases**:
- Training agents on Git operations
- Task-based development workflows
- Code review automation

**Optimization**: Fast resets (<1s) via task-based configuration

**Example Workflow**:
```python
from envs.git_env import GitEnv

client = GitEnv.from_docker_image("git-env:latest")
client.reset()

# List repos
repos = client.step(list_repos())

# Clone a repository
client.step(clone_repo("my-project"))

# Run git commands
client.step(execute_git_command("git status", "/workspace/my-project"))
```

---

### 9. FinRL Environment

**Path**: `src/envs/finrl_env/`

**Purpose**: Stock trading simulation via FinRL integration

**Action**:
```python
@dataclass
class FinRLAction(Action):
    actions: list[float]  # Buy/sell actions per stock
                          # Positive = buy, Negative = sell
```

**Observation**:
```python
@dataclass
class FinRLObservation(Observation):
    state: list[float]        # Market state
    portfolio_value: float    # Current portfolio value
    date: str                 # Trading date
```

**Features**:
- Technical indicators (MACD, RSI, CCI, DX)
- Configurable trading parameters
- Historical stock data
- Portfolio tracking

**Use Cases**:
- Algorithmic trading agent training
- Financial strategy optimization
- Market prediction research

---

### 10. SUMO-RL Environment

**Path**: `src/envs/sumo_rl_env/`

**Purpose**: Traffic signal control optimization

**Simulation**: SUMO (Simulation of Urban MObility)

**Features**:
- Multi-agent traffic coordination
- Configurable reward functions:
  - Waiting time minimization
  - Queue length reduction
  - Pressure balancing
  - Speed optimization

**Use Cases**:
- Smart city traffic optimization
- Multi-agent coordination research
- Urban planning simulation

---

### 11. TextArena Environment

**Path**: `src/envs/textarena_env/`

**Purpose**: Text-based games and puzzles

**Games Include**:
- Wordle
- GuessTheNumber
- Chess (text representation)
- Word puzzles
- Logic games

**Use Cases**:
- Language-based puzzle solving
- Text game playing
- Natural language strategy learning

**Configuration**: Select game via environment variables

---

### 12. DIPG Safety Environment

**Path**: `src/envs/dipg_safety_env/`

**Purpose**: Alignment and safety-focused tasks

**Use Cases**:
- AI safety research
- Alignment testing
- Safe behavior training

---

### Environment Summary Table

| Environment | Domain | Complexity | Best For |
|-------------|--------|------------|----------|
| **Echo** | Testing | Minimal | Learning framework |
| **OpenSpiel** | Games | Medium | Game AI, multi-agent |
| **Connect4** | Board Games | Low | Two-player strategy |
| **Atari** | Video Games | Medium | Visual RL, benchmarking |
| **BrowserGym** | Web Automation | High | Real-world task automation |
| **Coding** | Code Execution | Medium | LLM code generation |
| **Chat** | Conversation | Low | Dialogue systems |
| **Git** | Dev Tools | Medium | Code workflow automation |
| **FinRL** | Finance | High | Trading algorithms |
| **SUMO-RL** | Traffic Sim | High | Multi-agent coordination |
| **TextArena** | Text Games | Low-Medium | Language puzzles |
| **DIPG Safety** | AI Safety | Varies | Alignment research |

---

## How to Add a New Environment

OpenEnv makes it easy to create custom environments. Here's a complete guide from scaffolding to deployment.

### Quick Start

```bash
# 1. Create new environment from template
openenv init my_game_env

# 2. Navigate to environment
cd my_game_env

# 3. Implement your logic (see below)

# 4. Test locally
python test_my_game.py

# 5. Build Docker image
docker build -t my-game-env:latest -f server/Dockerfile .

# 6. Deploy to Hugging Face (optional)
openenv push --repo-id username/my-game-env
```

### Step-by-Step Guide

#### Step 1: Define Your Models

Edit `models.py` to define your action, observation, and state:

```python
# models.py
from dataclasses import dataclass
from core.env_server.types import Action, Observation, State

@dataclass
class GameAction(Action):
    """Actions the agent can take in your game"""
    move_direction: str  # "up", "down", "left", "right"
    use_item: bool = False

@dataclass
class GameObservation(Observation):
    """What the agent observes after each action"""
    player_position: tuple[int, int]
    health: int
    score: int
    game_over: bool = False
    message: str = ""

@dataclass
class GameState(State):
    """Custom episode state (inherits episode_id, step_count)"""
    level: int = 1
    high_score: int = 0
```

#### Step 2: Implement Environment Logic

Edit `server/game_environment.py`:

```python
# server/game_environment.py
import uuid
from core.env_server.interfaces import Environment
from ..models import GameAction, GameObservation, GameState

class GameEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = GameState()
        self._player_pos = (0, 0)
        self._health = 100
        self._score = 0

    def reset(self) -> GameObservation:
        """Start a new game episode"""
        self._state = GameState(episode_id=str(uuid.uuid4()))
        self._player_pos = (0, 0)
        self._health = 100
        self._score = 0

        return GameObservation(
            player_position=self._player_pos,
            health=self._health,
            score=self._score,
            message="Game started! Good luck!"
        )

    def step(self, action: GameAction) -> GameObservation:
        """Execute one game step"""
        self._state.step_count += 1

        # Process movement
        x, y = self._player_pos
        if action.move_direction == "up":
            y += 1
        elif action.move_direction == "down":
            y -= 1
        elif action.move_direction == "left":
            x -= 1
        elif action.move_direction == "right":
            x += 1

        self._player_pos = (x, y)

        # Process item usage
        if action.use_item:
            self._health = min(100, self._health + 10)
            message = "Used health potion!"
        else:
            message = f"Moved {action.move_direction}"

        # Update score
        self._score += 10

        # Check game over
        game_over = self._health <= 0

        return GameObservation(
            player_position=self._player_pos,
            health=self._health,
            score=self._score,
            game_over=game_over,
            message=message
        )

    @property
    def state(self) -> GameState:
        """Return current episode state"""
        return self._state
```

#### Step 3: Create FastAPI Server

Edit `server/app.py`:

```python
# server/app.py
from core.env_server import create_fastapi_app
from ..models import GameAction, GameObservation
from .game_environment import GameEnvironment

env = GameEnvironment()
app = create_fastapi_app(env, GameAction, GameObservation)

# This automatically creates:
# - POST /reset
# - POST /step
# - GET /state
# - GET /health
# - WebSocket /ws (for web interface)
```

#### Step 4: Define Dependencies

Create `server/requirements.txt`:

```txt
# Add any Python packages your environment needs
numpy>=1.24.0
pygame>=2.5.0
```

For complex setup, create `server/install_deps.sh`:

```bash
#!/bin/bash
set -e

# Install Python dependencies
pip install --no-cache-dir -r /tmp/requirements.txt

# Additional setup (only if needed)
apt-get update && apt-get install -y some-system-package
```

#### Step 5: Create Dockerfile

Edit `server/Dockerfile`:

```dockerfile
# Use OpenEnv base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY src/envs/my_game_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/my_game_env/ /app/src/envs/my_game_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "envs.my_game_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 6: Implement Client

Edit `client.py`:

```python
# client.py
from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult
from .models import GameAction, GameObservation, GameState

class GameEnv(HTTPEnvClient[GameAction, GameObservation]):
    """Client for interacting with GameEnvironment"""

    def _step_payload(self, action: GameAction) -> dict:
        """Convert action to JSON payload"""
        return {
            "move_direction": action.move_direction,
            "use_item": action.use_item
        }

    def _parse_result(self, payload: dict) -> StepResult[GameObservation]:
        """Parse server response into observation"""
        obs = GameObservation(
            player_position=tuple(payload["observation"]["player_position"]),
            health=payload["observation"]["health"],
            score=payload["observation"]["score"],
            game_over=payload["observation"].get("game_over", False),
            message=payload["observation"].get("message", "")
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> GameState:
        """Parse episode state"""
        return GameState(**payload)
```

#### Step 7: Update Package Exports

Edit `__init__.py`:

```python
# __init__.py
from .models import GameAction, GameObservation, GameState
from .client import GameEnv

__all__ = ["GameAction", "GameObservation", "GameState", "GameEnv"]
```

#### Step 8: Create openenv.yaml

Create `openenv.yaml` for Hugging Face deployment:

```yaml
spec_version: 1
name: my_game_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

#### Step 9: Write Documentation

Edit `README.md`:

```markdown
# My Game Environment

A custom OpenEnv environment for training agents to play MyGame.

## Quick Start

```python
from envs.my_game_env import GameAction, GameEnv

# Start environment
client = GameEnv.from_docker_image("my-game-env:latest")

# Reset
result = client.reset()
print(result.observation.message)  # "Game started! Good luck!"

# Take actions
result = client.step(GameAction(move_direction="right", use_item=False))
print(result.observation.score)  # 10

# Cleanup
client.close()
```

## Action Space

- `move_direction`: "up", "down", "left", "right"
- `use_item`: Boolean, use health potion

## Observation Space

- `player_position`: (x, y) coordinates
- `health`: Integer, 0-100
- `score`: Integer, cumulative score
- `game_over`: Boolean
- `message`: String, status message

## Building

```bash
# Build base image first
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build environment
docker build -t my-game-env:latest -f src/envs/my_game_env/server/Dockerfile .
```
```

#### Step 10: Test Your Environment

Create `test_my_game.py`:

```python
# test_my_game.py
from envs.my_game_env.server.game_environment import GameEnvironment
from envs.my_game_env.models import GameAction

def test_environment():
    env = GameEnvironment()

    # Test reset
    obs = env.reset()
    assert obs.health == 100
    assert obs.score == 0
    print("✓ Reset works")

    # Test movement
    obs = env.step(GameAction(move_direction="right"))
    assert env.state.step_count == 1
    print("✓ Step works")

    # Test item usage
    obs = env.step(GameAction(move_direction="up", use_item=True))
    assert "potion" in obs.message.lower()
    print("✓ Item usage works")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_environment()
```

Run tests:
```bash
python test_my_game.py
```

#### Step 11: Build Docker Images

```bash
# Build base image (if not already built)
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build your environment
docker build -t my-game-env:latest -f src/envs/my_game_env/server/Dockerfile .
```

#### Step 12: Test with Docker

```python
from envs.my_game_env import GameAction, GameEnv

# This will automatically start the Docker container
client = GameEnv.from_docker_image("my-game-env:latest")

# Test
result = client.reset()
print(f"Health: {result.observation.health}")

result = client.step(GameAction(move_direction="right"))
print(f"Position: {result.observation.player_position}")
print(f"Score: {result.observation.score}")

client.close()  # Stops and removes container
```

#### Step 13: Add to GitHub Actions (Optional)

Edit `.github/workflows/docker-build.yml`:

```yaml
strategy:
  matrix:
    image:
      # ... existing environments ...
      - name: my-game-env
        dockerfile: src/envs/my_game_env/server/Dockerfile
```

This enables automatic Docker builds on every push to `main`.

#### Step 14: Deploy to Hugging Face (Optional)

```bash
cd src/envs/my_game_env
openenv push --repo-id your-username/my-game-env
```

This will:
1. Validate `openenv.yaml`
2. Enable web interface automatically
3. Deploy to Hugging Face Spaces
4. Return the public URL

### Best Practices

#### 1. Type Safety

Always use explicit types:

```python
@dataclass
class MyAction(Action):
    value: int  # Good: Explicit type
    # value  # Bad: No type annotation
```

#### 2. Error Handling

Handle errors gracefully:

```python
def step(self, action: MyAction) -> MyObservation:
    try:
        result = self._process_action(action)
        return MyObservation(result=result, success=True)
    except ValueError as e:
        return MyObservation(
            result="",
            success=False,
            error_message=str(e)
        )
```

#### 3. State Management

Track all relevant episode data:

```python
@dataclass
class MyState(State):
    # Custom fields
    total_reward: float = 0.0
    actions_taken: list[str] = field(default_factory=list)

    # State automatically includes:
    # - episode_id: str
    # - step_count: int
    # - start_time: datetime
```

#### 4. Documentation

Include clear documentation:

- Overview and purpose
- Action/Observation specifications
- Quick start example
- Build instructions
- Dependencies list

#### 5. Minimal Dependencies

Only include necessary packages in `requirements.txt`:

```txt
# Good: Only what you need
numpy>=1.24.0

# Bad: Unnecessary dependencies
numpy>=1.24.0
pandas  # Not used
tensorflow  # Not used
```

---

## Integration Examples

### Using OpenEnv with torchforge (GRPO)

```python
# From examples/grpo_blackjack/
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction
import torch
from torchforge import GRPO

# Create environment
env = OpenSpielEnv.from_docker_image(
    "openspiel-env:latest",
    game="blackjack"
)

# Training loop
for episode in range(1000):
    obs = env.reset()
    done = False

    while not done:
        # Get action from policy
        action = policy(obs)

        # Execute action
        result = env.step(OpenSpielAction(action=action))

        # Update policy with GRPO
        grpo.update(result.observation, result.reward)

        done = result.done

env.close()
```

### Using OpenEnv with TRL

```python
from envs.coding_env import CodingEnv, CodeAction
from trl import GRPOTrainer

env = CodingEnv.from_docker_image("coding-env:latest")

trainer = GRPOTrainer(
    model=model,
    env=env,
    config=config
)

trainer.train()
```

### Using OpenEnv with Custom Framework

```python
from envs.browsergym_env import BrowserGymEnv, BrowserAction

env = BrowserGymEnv.from_docker_image("browsergym-env:latest")

# Your custom training loop
for epoch in range(num_epochs):
    obs = env.reset()

    for step in range(max_steps):
        # Your policy
        action_str = your_model.predict(obs)

        # Execute
        result = env.step(BrowserAction(action=action_str))

        # Your learning update
        your_model.update(result.observation, result.reward)

        if result.done:
            break

env.close()
```

---

## Resources and Community

### Official Documentation

- **Main README**: [GitHub](https://github.com/meta-pytorch/OpenEnv)
- **Core Framework**: `src/core/README.md`
- **Environment Building Guide**: `src/envs/README.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`

### Tutorials and Examples

- **Interactive Tutorial**: [Google Colab Notebook](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
- **GRPO BlackJack Example**: `examples/grpo_blackjack/`
- **Local Examples**: `examples/` directory (18+ scripts)

### Community and Support

- **Discord**: [OpenEnv Community](https://discord.gg/YsTYBh6PD9)
- **GitHub Issues**: [Report bugs and request features](https://github.com/meta-pytorch/OpenEnv/issues)
- **PyPI Package**: `pip install openenv-core`

### Integration Documentation

- **TRL Integration**: [Hugging Face Docs](https://huggingface.co/docs/trl/main/en/openenv)
- **Unsloth Example**: [Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)
- **SkyRL Integration**: [SkyRL Docs](https://skyrl.readthedocs.io/en/latest/examples/openenv.html)
- **ART Integration**: [OpenPipe ART](https://art.openpipe.ai/integrations/openenv-integration)

### Deployment Platforms

- **Hugging Face Spaces**: Deploy with `openenv push`
- **Lightning AI Studio**: [Featured Environments](https://lightning.ai/environments?section=featured)
- **GitHub Container Registry**: Pre-built images at `ghcr.io/meta-pytorch/`

### Requirements

- **Python**: 3.11+
- **Docker**: Desktop or Engine
- **Dependencies**: FastAPI, Pydantic, Uvicorn, Requests (auto-installed with `openenv-core`)

### Supporters and Contributors

OpenEnv is supported by:
- Meta PyTorch (core team)
- Hugging Face
- Patronus AI
- Surge AI
- LastMile AI
- Unsloth AI
- vLLM
- SkyRL (UC-Berkeley)
- Lightning AI
- Stanford Scaling Intelligence Lab
- Fleet AI
- And growing community contributors

### License

BSD 3-Clause License - See LICENSE file

---

## Summary

OpenEnv is a powerful, standardized framework for agentic RL training that:

1. **Solves Real Problems**: Unified API, containerization, easy deployment
2. **Rich Ecosystem**: 70+ environments from games to web automation to trading
3. **Production-Ready**: Used by major RL frameworks and research institutions
4. **Easy to Extend**: CLI scaffolding, templates, and comprehensive guides
5. **Community-Driven**: Active development with growing contributor base

Whether you're:
- **Training agents** on diverse tasks
- **Creating new environments** for specific domains
- **Building RL frameworks** that need standardized environments

OpenEnv provides the tools, documentation, and community to help you succeed.

### Next Steps

1. **Try the Tutorial**: [Google Colab Notebook](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
2. **Explore Examples**: `examples/` directory
3. **Create Your Environment**: `openenv init my_env`
4. **Join Community**: [Discord](https://discord.gg/YsTYBh6PD9)
5. **Contribute**: [GitHub](https://github.com/meta-pytorch/OpenEnv)

Happy coding!
