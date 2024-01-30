import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
import os
# from simple_env import SimpleEnv
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from minigrid.wrappers import ImgObsWrapper

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True

# 自定义的特征提取器
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

# policy_kwargs = dict(
# #     features_extractor="mlp",
# )

model_dir = "models/DoorKey1"
monitor_dir = "monitor_logs/DoorKey1/"

env = gym.make("MiniGrid-DoorKey-6x6-v0") # 训练时记得关掉渲染
env = Monitor(env, filename=monitor_dir, allow_early_resets=True)
env = ImgObsWrapper(env)

def train():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=monitor_dir)
    TIMESTEPS = 1e6 
    model.learn(total_timesteps=int(TIMESTEPS), callback=callback)
    plt.show()
    # iters = 0
    # while iters < 10:
    #     iters += 1
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    #     model.save(f"{model_dir}/{TIMESTEPS*iters}") # 10000个时间步存储一次模型
        
def test(model_name="10000.zip"):
    model_path = f"{model_dir}" + "/" + model_name
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
    train()
    
    #test()
