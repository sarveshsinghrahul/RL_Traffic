import os
import sys
import numpy as np
import gym
from gym import spaces
import traci

# Ensure SUMO_HOME/tools is in sys.path
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set SUMO_HOME environment variable")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools not in sys.path:
    sys.path.append(tools)


class SumoTrafficEnv(gym.Env):
    """
    RL environment for a single SUMO intersection.

    - Observation: [queue_lane_0, ..., queue_lane_N-1, current_phase]
    - Action: phase index (0 .. num_phases-1)
    - Reward: - (total halting vehicles on controlled lanes)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sumocfg_path: str = "intersection.sumocfg",
        gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 5,
    ):
        super().__init__()

        self.sumocfg_path = sumocfg_path
        self.gui = gui
        self.max_steps = max_steps       # seconds of simulation per episode
        self.delta_time = delta_time     # seconds per RL step

        self.step_count = 0
        self._sumo_started = False

        # discover TLS, controlled lanes, and number of phases (headless)
        self._discover_network()

        # ---- Gym spaces ----
        self.n_lanes = len(self.controlled_lanes)
        self.action_space = spaces.Discrete(self.num_phases)

        # obs: queue for each lane + current phase
        low = np.zeros(self.n_lanes + 1, dtype=np.float32)
        high = np.full(self.n_lanes + 1, 100.0, dtype=np.float32)  # max queue guess
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ---------- SUMO helpers ---------- #

    def _start_sumo(self, use_gui: bool | None = None):
        if use_gui is None:
            use_gui = self.gui
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        traci.start(
            [sumo_binary, "-c", self.sumocfg_path, "--no-step-log", "true", "--start"]
        )
        self._sumo_started = True

    def _close_sumo(self):
        if self._sumo_started and traci.isLoaded():
            traci.close()
        self._sumo_started = False

    def _discover_network(self):
        """Start SUMO once (headless) to get TLS id, controlled lanes, and num phases."""
        # always headless for discovery so we don't open a GUI here
        self._start_sumo(use_gui=False)

        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            self._close_sumo()
            raise RuntimeError("No traffic lights found in the network.")

        self.tl_id = tls_ids[0]
        print("[ENV] TLS:", self.tl_id)

        # controlled lanes (deduplicated)
        raw_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        seen = set()
        self.controlled_lanes = []
        for l in raw_lanes:
            if l not in seen:
                seen.add(l)
                self.controlled_lanes.append(l)

        print("[ENV] Controlled lanes:", self.controlled_lanes)

        # number of phases
        prog_defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)
        phases = prog_defs[0].phases
        self.num_phases = len(phases)
        print("[ENV] Number of phases:", self.num_phases)

        self._close_sumo()

    # ---------- Gym API ---------- #

    def reset(self):
        """Start a fresh SUMO episode and return initial state."""
        self._close_sumo()
        self._start_sumo()   # now uses self.gui (so gui=True will show window)
        self.step_count = 0
        state = self._get_state()
        return state

    def _get_state(self):
        # queues on each controlled lane
        queues = []
        for lane in self.controlled_lanes:
            q = traci.lane.getLastStepHaltingNumber(lane)
            queues.append(float(q))

        # current phase index
        phase = float(traci.trafficlight.getPhase(self.tl_id))
        state = np.array(queues + [phase], dtype=np.float32)
        return state

    def step(self, action):
        """Apply chosen phase, step SUMO, return (state, reward, done, info)."""
        self.step_count += 1

        # clip / wrap action to valid phase index
        phase_idx = int(action) % self.num_phases
        traci.trafficlight.setPhase(self.tl_id, phase_idx)

        # advance simulation for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()

        # new state
        state = self._get_state()

        # reward = - total halting vehicles on controlled lanes
        total_halts = 0.0
        for lane in self.controlled_lanes:
            total_halts += traci.lane.getLastStepHaltingNumber(lane)
        reward = -total_halts

        # episode termination
        done = False
        if self.step_count * self.delta_time >= self.max_steps:
            done = True
        if traci.simulation.getMinExpectedNumber() == 0:
            done = True

        info = {"total_halts": total_halts}

        if done:
            self._close_sumo()

        return state, float(reward), done, info

    def render(self, mode="human"):
        # gui=True already shows the GUI
        pass

    def close(self):
        self._close_sumo()
