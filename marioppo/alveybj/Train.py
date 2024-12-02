import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack

log_dir = "logs/"
env_id = "ALE/Breakout-v5"
# env_id = "CartPole-v1"
save_dir = "lstm_models"

# learning_rate=3e-5, ent_coef=1e-2

def make_env(env_id, seed, rank, render_mode):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        env.seed(seed + rank)  # Ensure different seeds for different processes
        env = Monitor(env, log_dir)
        return env
    return _init
    
if __name__ == "__main__":
    num_cpu = 16
    vec_env = SubprocVecEnv([make_env(env_id, 0, i, "rgb_array") for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./tensorboard_logs/")
    # model = RecurrentPPO("CnnLstmPolicy", vec_env, verbose=1, tensorboard_log="./tesnorboard_logs/")
    eval_callback = EvalCallback(vec_env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=1000, deterministic=True, render=False)
    model.learn(total_timesteps=1e7, callback = eval_callback)