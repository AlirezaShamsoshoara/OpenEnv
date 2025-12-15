# Doom Environment for OpenEnv: Visual RL Training with a Classic

## Project Overview

Integration of **ViZDoom** into **OpenEnv** for training vision-based agents using reinforcement learning. This environment enables training CNNs, Vision Transformers (ViTs), and Vision-Language Models (VLMs) on first-person visual navigation and combat tasks.

**Resources**:
- **GitHub**: https://github.com/AlirezaShamsoshoara/OpenEnv/tree/alidev_doom_env_01
- **HuggingFace Space**: https://huggingface.co/spaces/Crashbandicoote2/doom_env
- **ViZDoom GitHub**: https://github.com/Farama-Foundation/ViZDoom
- **ViZDoom Docs**: http://vizdoom.cs.put.edu.pl/

---

## Why Doom?

Doom (1993) is more than just a legendary game—it's the perfect testbed for visual RL:

- **Nostalgia Factor**: Everyone remembers playing Doom. It's a game that resonates across generations of developers and researchers alike.
- **Visual Complexity**: First-person 3D environments with dynamic lighting, textures, and moving entities provide rich visual input for CNNs and ViTs.
- **Action-Oriented Tasks**: Combat, navigation, and survival mechanics create diverse RL challenges with clear reward signals.
- **Lightweight**: Unlike modern games, Doom runs efficiently even without GPU, making it ideal for large-scale distributed training.
- **Established Research Platform**: ViZDoom is a well-known benchmark in the RL community with years of research behind it.

---

## Technical Highlights

### Production-Ready API

```python
from envs.doom_env import DoomAction, DoomEnv

# Deploy with Docker - fully isolated
env = DoomEnv.from_docker_image("doom-env:latest")

# Standard RL interface
obs = env.reset()
print(f"Screen shape: {obs.observation.screen_shape}")  # e.g., [240, 320, 3]

# Take actions and get visual observations
for step in range(1000):
    obs = env.step(DoomAction(action_id=1))  # Move, shoot, etc.
    frame = np.array(obs.observation.screen_buffer).reshape(obs.observation.screen_shape)
    # Feed frame to CNN/ViT/VLM

env.close()
```

### Key Features

• Visual Observations: RGB or grayscale screen buffers (configurable resolution)
• Game Variables: Health, ammo, kills, armor - perfect for reward shaping
• Multiple Scenarios: basic, deadly_corridor, defend_the_center, health_gathering, my_way_home, predict_position, take_cover
• Flexible Actions: Discrete action IDs or button combinations
• Docker Deployment: Fully containerized with web interface for debugging
• Web UI: Interactive browser-based visualization at /web

### Vision Model Integration

The environment is designed for vision-based learning:

```python
# Process observations for vision models
obs = env.reset()
screen = np.array(obs.observation.screen_buffer).reshape(obs.observation.screen_shape)

# For CNN: Direct pixel input [H, W, C]
cnn_input = torch.tensor(screen).permute(2, 0, 1).float() / 255.0

# For ViT: Patch-based processing
vit_input = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(screen)

# For VLM: Combine with task description
task_prompt = "Navigate the corridor and eliminate enemies"
vlm_input = (screen, task_prompt)
```

---

## Scenarios for Diverse RL Challenges

- **Basic**: Learn fundamental movement and shooting
- **Deadly Corridor**: Navigate while avoiding/eliminating enemies
- **Defend the Center**: Survive waves of attacking monsters
- **Health Gathering**: Resource collection under time pressure
- **My Way Home**: Pure navigation to find an exit
- **Predict Position**: Anticipate and hit moving targets
- **Take Cover**: Learn defensive strategies

Each scenario provides unique challenges for visual RL: spatial reasoning, temporal planning, resource management, and threat assessment.

---

## Use Cases

1. **CNN Feature Learning**: Train convolutional networks to extract meaningful features from first-person game frames
2. **ViT for Visual RL**: Apply Vision Transformers to understand spatial relationships in 3D environments
3. **VLM Agents**: Use Vision-Language Models to follow natural language instructions ("Find the exit", "Collect health packs")
4. **Hierarchical RL**: High-level planning (explore, fight, flee) + low-level control (movement, aiming)
5. **Curriculum Learning**: Progress from basic scenarios to complex multi-objective tasks

---

## Quick Start

```bash
# Clone and checkout the branch
git clone -b alidev_doom_env_01 https://github.com/AlirezaShamsoshoara/OpenEnv.git
cd OpenEnv

# Pull the Docker image
docker pull vmvm-registry.fbinfra.net/hackathon/pe_hackathon/alisol/doom-env:latest

# Run example
python examples/doom_example.py --docker --render

# Or access web interface
docker run -p 8000:8000 vmvm-registry.fbinfra.net/hackathon/pe_hackathon/alisol/doom-env:latest
# Open http://localhost:8000/web
```

---

**Bottom Line**: Doom Environment brings a timeless classic into OpenEnv, enabling researchers to train vision models on engaging first-person RL tasks. It combines nostalgia with cutting-edge AI research—because training agents is more fun when they're fighting demons.
