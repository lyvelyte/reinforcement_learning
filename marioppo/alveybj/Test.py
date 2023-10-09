import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

model = RecurrentPPO.load(r"models/best_model.zip")
# model = PPO.load(r"models/best_model.zip")

def make_env(env_id: str, render_mode: str):
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=True, action_repeat_probability=0.0)
    env.reset()
    return env 

def main():
    env_id = "ALE/Breakout-v5"
    # env_id = "CartPole-v1"
    
    env =  make_env(env_id, "human")
    obs, rew, done, trunc, info = env.step(env.action_space.sample())
    done = False
    while not done:
        action, state = model.predict(obs)
        obs, rew, done, trunc, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    main()