import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv

# model = RecurrentPPO.load(r"lstm_models/best_model.zip")
model = PPO.load(r"lstm_models/best_model.zip")

def make_env(env_id, seed, rank, render_mode):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        env.metadata['render_fps'] = 30
        env.seed(seed + rank)  # Ensure different seeds for different processes
        return env
    return _init

def main():
    env_id = "ALE/Breakout-v5"
    # env_id = "CartPole-v1"
    
    num_cpu = 1
    vec_env = SubprocVecEnv([make_env(env_id, 0, i, "human") for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    obs, rew, done, trunc, info = vec_env.step(vec_env.action_space.sample())
    done = False
    while not done:
        action, state = model.predict(obs)
        obs, rew, done, trunc, info = vec_env.step(action)
        vec_env.render()
    vec_env.close()

if __name__ == "__main__":
    main()