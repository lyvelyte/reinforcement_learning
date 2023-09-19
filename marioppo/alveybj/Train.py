import retro
import gymnasium as gym
# import gym as old_gym
from gymnasium.wrappers import StepAPICompatibility
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import RecordEpisodeStatistics, AtariPreprocessing

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
  

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback to save the best model based on training reward.

    The check is performed every `check_freq` steps based on the training reward.
    It is recommended to use `EvalCallback` for practical usage.

    Parameters:
        check_freq (int): Frequency of checks.
        log_dir (str): Directory where the model and log will be saved.
        verbose (int, optional): Verbosity level. Defaults to 1.
    """
    
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.reward_buffer = np.zeros((10000, 1))
        self.step_cnt = 0

    def _init_callback(self) -> None:
        """Create directory for saving models if it does not exist."""
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Save the best model based on the mean reward of the last 100 episodes."""
        self.reward_buffer[self.step_cnt % len(self.reward_buffer)] = np.sum(self.locals.get('rewards', None))
        self.step_cnt = self.step_cnt + 1

        if self.n_calls % self.check_freq == 0:
            if np.mean(self.reward_buffer) > 0:
                mean_reward = np.mean(self.reward_buffer)
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array', max_episode_steps=10000)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = RecordEpisodeStatistics(env)
    return env


def main():
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    # Instantiate the agent with a PPO with LSTM policy
    env = make_env() 
    # model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, device="cuda", tensorboard_log="./board/", learning_rate=3e-5, ent_coef=1e-2)
    model = PPO('CnnPolicy', env, verbose=1, device="cuda", tensorboard_log="./board/", learning_rate=3e-5, ent_coef=1e-2)
    
    # Train the agent
    print("Training mario LSTM agent...")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=2e6, callback = callback, tb_log_name="PPO-LSTM-00001")

    # Save the trained model
    model.save("ppo_breakout")

if __name__ == "__main__":
    main()
