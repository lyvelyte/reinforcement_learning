import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def main():
    env = gym.make('SuperMarioBrosRandomStages-v0', stages=['1-1'], render_mode="human", apply_api_compatibility=True)
    env.reset()
    done = False
    while not done:
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        env.render()
    env.close()

if __name__ == "__main__":
    main()