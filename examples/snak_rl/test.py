from envs.snake_env import SnakeAction, SnakeEnv

# Connect to running server
client = SnakeEnv(base_url="http://localhost:8000")
result = client.reset()
result = client.step(SnakeAction(action=0))


# import gym
# import marlenv
# from marlenv.wrappers import SingleAgent

# env = gym.make("Snake-v1", num_snakes=1)
# env = SingleAgent(env)

# env, observation_space, action_space, properties = marlenv.wrappers.make_snake(
#     num_envs=1,  # Number of environments. Used to decided vector env or not
#     num_snakes=1,  # Number of players. Used to determine single/multi agent
# )


# import gym

# import marlenv.envs  # Import to register environments

# print("Registered environments with 'Snake' in name:")
# for env_name in gym.envs.registry.env_specs.keys():
#     if "Snake" in env_name or "snake" in env_name:
#         pass
#         # print(f"  - {env_name}")
#     print(f"  - {env_name}")
