from rl_env import SumoTrafficEnv

def main():
    env = SumoTrafficEnv(
        sumocfg_path="intersection.sumocfg",
        gui=True,      # show GUI so you can watch it
        max_steps=200,
        delta_time=2,  # 2s between decisions
    )

    obs = env.reset()
    print("Initial obs:", obs)

    for t in range(50):
        # random action to test env
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"t={t}, action={action}, reward={reward}, info={info}")
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
