import retro

def main():
    env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
    env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

if __name__ == "__main__":
    main()