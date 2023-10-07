import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

def make_env(env_id: str, render_mode: str):
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()
    env = AtariWrapper(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True, clip_reward=True, action_repeat_probability=0.0)
    return env 

if __name__ == "__main__":
    env_id = "ALE/Breakout-v5"
    # env_id = "CartPole-v1"
    num_cpu = 18
    vec_env = SubprocVecEnv([make_env(env_id, i, "rgb_array") for i in range(num_cpu)])
    # model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./board/", learning_rate=3e-5, ent_coef=1e-2)
    model = RecurrentPPO("CnnLstmPolicy", vec_env, verbose=1, tensorboard_log="./board/", learning_rate=3e-5, ent_coef=1e-2)
    model.learn(total_timesteps=2e5)
    model.save("ppo_breakout_model")