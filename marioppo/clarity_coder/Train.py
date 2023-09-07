import os
import numpy as np
import retro
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from RandomAgent import TimeLimitWrapper


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

    def _init_callback(self) -> None:
        """Create directory for saving models if it does not exist."""
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Save the best model based on the mean reward of the last 100 episodes."""
        if self.n_calls % self.check_freq == 0:
            x, y = results_plotter.ts2xy(results_plotter.load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def make_env(env_id, rank, seed=0):
    """Create an environment with various wrappers and a given seed.
    
    Parameters:
        env_id (str): The environment ID.
        rank (int): Index of the subprocess.
        seed (int, optional): Initial seed for RNG. Defaults to 0.
    """
    def _init():
        env = retro.make(game=env_id)
        env = TimeLimitWrapper(env, max_steps=4000)
        env = MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), log_dir)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=3e-5, 
                n_steps=2048, batch_size=16, n_epochs=1, gamma=0.99, gae_lambda=0.95, clip_range=0.3)
    
    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=5000000, callback=callback, tb_log_name="PPO-00001")
    model.save(env_id)
    
    print("------------- Done Learning -------------")
    
    env = retro.make(game=env_id)
    env = TimeLimitWrapper(env)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
