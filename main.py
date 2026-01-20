import numpy as np
import matplotlib.pyplot as plt

from ebm_sir import euler_sir
from abm_sir import abm_sir

def main():
    beta = 0.35
    gamma = 0.08

    dt = 0.1
    steps = 300

    # EBM
    t, S, I, R = euler_sir(
        beta=beta,
        gamma=gamma,
        S0=0.99,
        I0=0.01,
        R0=0.0,
        dt=dt,
        steps=steps
    )

    # ABM
    S2, I2, R2, pos, state = abm_sir(
        n=200,
        beta=beta,
        gamma=gamma,
        steps=steps,
        move_sigma=0.02,
        infection_radius=0.05,
        seed=42
    )

    # Plot comparison
    plt.figure()
    plt.plot(t, I, label="EBM Infected (Euler)")
    plt.plot(t, I2, label="ABM Infected (Agents)")
    plt.xlabel("Time")
    plt.ylabel("Fraction Infected")
    plt.title("SIR: EBM vs ABM")
    plt.legend()
    plt.show()

    # Final ABM state
    plt.figure()
    colors = np.array(["tab:blue", "tab:red", "tab:green"])
    plt.scatter(pos[:, 0], pos[:, 1], c=colors[state], s=20)
    plt.title("ABM Final State (Blue=S, Red=I, Green=R)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# THIS PART IS THE "RUN" PART — IT MUST BE HERE
if __name__ == "__main__":
    main()

