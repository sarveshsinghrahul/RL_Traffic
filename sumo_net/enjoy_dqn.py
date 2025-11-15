from stable_baselines3 import DQN
from rl_env import SumoTrafficEnv

def main():
    env = SumoTrafficEnv(
        sumocfg_path="intersection.sumocfg",
        gui=True,
        max_steps=6000,
        delta_time=2,
    )

    model = DQN.load("../models/dqn_sumo_traffic")

    obs = env.reset()
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step += 1
        if step % 20 == 0:
            print(f"step={step}, reward={reward}, info={info}")

    env.close()

if __name__ == "__main__":
    main()
