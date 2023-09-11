import retro
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.policies import LstmPolicy

def main():
    env = gym.make(game='SuperMarioBros-Nes', state='Level1-1')

    # VecEnv is used as Stable Baselines3 algorithms expect vectorized environment
    env = DummyVecEnv([lambda: env])

    # Instantiate the agent with a PPO with LSTM policy
    model = PPO(LstmPolicy, env, verbose=1, policy_kwargs={'net_arch': [64, 'lstm', dict(vf=[64], pi=[64])]})
    
    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_lstm_mario")

    # Test the trained agent
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    main()
