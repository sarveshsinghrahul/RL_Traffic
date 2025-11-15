from stable_baselines3 import DQN
from rl_env import SumoTrafficEnv
import os

def main():
    env = SumoTrafficEnv(
        sumocfg_path="intersection.sumocfg",
        gui=False,      # no GUI during training (faster)
        max_steps=3600, # 1 hour of sim per episode
        delta_time=5,   # agent acts every 5s
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
    )

    model.learn(total_timesteps=200_000)

    os.makedirs("../models", exist_ok=True)
    model.save("../models/dqn_sumo_traffic")

    env.close()
    print("Training complete. Model saved to ../models/dqn_sumo_traffic.zip")

if __name__ == "__main__":
    main()
