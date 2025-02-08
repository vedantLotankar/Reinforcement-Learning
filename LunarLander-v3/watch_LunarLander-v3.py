import gymnasium as gym
from stable_baselines3 import DQN

model = DQN.load("Reinforcement-Learning/LunarLander-v3/dqn_lunar_lander_v3")

env = gym.make("LunarLander-v3", render_mode = "human")

# Reset the environment to get the initial observation
obs, _ = env.reset()

# Run the simulation
for _ in range(500):  # 500 steps (adjust if you want to watch longer!)
    action, _ = model.predict(obs, deterministic=True)  # Get action from model
    obs, reward, done, _, _ = env.step(action)  # Apply action to the environment
    env.render()  # Visualize the environment
    
    if done:
        obs, _ = env.reset()  # Reset environment if done (just in case)

env.close()  # Close the environment when done