import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
# from simple_env import SimpleEnv
import torch
from torch import nn
from minigrid.wrappers import ImgObsWrapper

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
    
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human") # 训练时记得关掉渲染
env = ImgObsWrapper(env)

# env = SimpleEnv(size=10, agent_start_pos=(1, 1), agent_start_dir=0, max_steps=1000)
# env.reset()
model_dir = "models/DoorKey1"

def train():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    TIMESTEPS = 100000
    iters = 0

    while iters < 10:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{TIMESTEPS*iters}") # 100000个时间步存储一次模型
        
def test():
    model_path = f"{model_dir}/100000.zip"
    model = PPO.load(model_path, env=env)
    vec_env = model.get_env()
    
    episodes = 10
    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            action, states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            env.render()
            print(rewards)
            
if __name__ == '__main__':
    # train()
    test()

# episodes = 10
# vec_env = model.get_env()
# obs = vec_env.reset()

# for ep in range(episodes):
#     done = False
#     while not done:
#         action, states = model.predict(obs)
#         obs, rewards, done,   info = vec_env.step(action)
#         env.render()
#         print(rewards)