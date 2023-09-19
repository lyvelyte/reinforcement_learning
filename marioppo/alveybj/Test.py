import gym
from sb3_contrib import RecurrentPPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv

# model = RecurrentPPO.load("tmp/best_model.zip")
model = PPO.load("tmp/best_model.zip")

def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode='human', max_episode_steps=10000)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def main():
    # env = gym.make('SuperMarioBrosRandomStages-v0', stages=['1-1'], render_mode="human", apply_api_compatibility=True)
    env =  make_env()
    env.reset()
    obs, rew, done, trunc, info = env.step(env.action_space.sample())
    done = False
    while not done:
        action, state = model.predict(obs)
        obs, rew, done, trunc, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    main()