import os
import sys
import traci

# 1. Make sure SUMO_HOME/tools is in sys.path
if "SUMO_HOME" not in os.environ:
    raise SystemExit("Please set SUMO_HOME environment variable")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools not in sys.path:
    sys.path.append(tools)


def main():
    # 2. Choose SUMO binary and config file
    sumo_binary = "sumo-gui"         # or "sumo" if you don't want GUI
    config = "intersection.sumocfg"  # assuming this file is in the same folder

    # 3. Start SUMO with TraCI
    traci.start([sumo_binary, "-c", config, "--no-step-log", "true", "--start"])
    print("Connected to SUMO")

    # 4. Get traffic light IDs
    tls_ids = traci.trafficlight.getIDList()
    print("Traffic lights:", tls_ids)
    if not tls_ids:
        print("No TLS found, exiting.")
        traci.close()
        return

    tls_id = tls_ids[0]
    print("Using TLS:", tls_id)

    # 5. Get controlled lanes (deduplicated)
    controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
    print("Controlled lanes:", controlled_lanes)

    # 6. Get number of phases from TLS program
    prog_defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    phases = prog_defs[0].phases  # assume one program
    num_phases = len(phases)
    print("Number of phases:", num_phases)

    # 7. Step the simulation, switch phase every 40 steps
    for step in range(200):
        if step % 40 == 0:
            phase = traci.trafficlight.getPhase(tls_id)
            next_phase = (phase + 1) % num_phases
            print(f"[step {step}] switching phase {phase} -> {next_phase}")
            traci.trafficlight.setPhase(tls_id, next_phase)

        total_halts = 0
        for lane in controlled_lanes:
            total_halts += traci.lane.getLastStepHaltingNumber(lane)

        print(f"[step {step}] total_halts = {total_halts}")
        traci.simulationStep()

    traci.close()
    print("Simulation finished and connection closed.")


if __name__ == "__main__":
    main()
