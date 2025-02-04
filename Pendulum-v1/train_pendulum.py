import gym
import torch
from stable_baselines3 import PPO

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create the environment
env = gym.make("Pendulum-v1", render_mode="human")  # Render mode enables visualization

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the model
model.learn(total_timesteps=100000)  # Training starts

# Save the model
model.save("ppo_pendulum")

print("Training completed. Now running the trained model...")

# Run the trained model and visualize
obs, _ = env.reset()
for _ in range(500):  # Run for 500 steps
    action, _ = model.predict(obs, deterministic=True)  # Get action from model
    obs, reward, done, _, _ = env.step(action)  # Apply action
    env.render()  # Show visualization

env.close()
