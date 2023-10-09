import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os 
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


log_dir = "logs/"
env_id = "ALE/Breakout-v5"
# env_id = "CartPole-v1"

save_dir = "models"

def make_env(env_id, seed, rank, render_mode):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True, clip_reward=True, action_repeat_probability=0.0)
        env.seed(seed + rank)  # Ensure different seeds for different processes
        env = Monitor(env, log_dir)
        return env
    return _init
    
if __name__ == "__main__":
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(env_id, 0, i, "rgb_array") for i in range(num_cpu)])
    # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./board/", learning_rate=3e-5, ent_coef=1e-2)
    model = RecurrentPPO("CnnLstmPolicy", vec_env, verbose=1, tensorboard_log="./board/")
    eval_callback = EvalCallback(vec_env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=500,deterministic=True, render=False)
    model.learn(total_timesteps=2e6, callback = eval_callback)