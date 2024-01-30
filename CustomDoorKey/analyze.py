
from stable_baselines3.common.results_plotter import plot_results, load_results
import stable_baselines3.common.results_plotter as results_plotter

from minigrid.wrappers import ImgObsWrapper
from matplotlib import pyplot as plt


monitor_dir = "monitor_logs/DoorKey1/"
timesteps = 1e6
plot_results([monitor_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO MiniGrid")
plt.show()