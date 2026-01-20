import numpy as np

def abm_sir(
    n: int = 200,
    beta: float = 0.35,
    gamma: float = 0.08,
    steps: int = 300,
    move_sigma: float = 0.02,
    infection_radius: float = 0.05,
    seed: int = 42
):
    """
    Simple 2D Agent-Based SIR model.

    States:
      0 = Susceptible (S)
      1 = Infected (I)
      2 = Recovered (R)
    """
    rng = np.random.default_rng(seed)

    pos = rng.random((n, 2))
    state = np.zeros(n, dtype=int)
    state[0] = 1  # start with 1 infected

    S_hist = np.zeros(steps + 1, dtype=float)
    I_hist = np.zeros(steps + 1, dtype=float)
    R_hist = np.zeros(steps + 1, dtype=float)

    def record(k: int):
        S_hist[k] = np.mean(state == 0)
        I_hist[k] = np.mean(state == 1)
        R_hist[k] = np.mean(state == 2)

    record(0)
    r2 = infection_radius ** 2

    for k in range(1, steps + 1):
        pos += rng.normal(0.0, move_sigma, size=pos.shape)
        pos = np.clip(pos, 0.0, 1.0)

        infected_idx = np.where(state == 1)[0]
        susceptible_idx = np.where(state == 0)[0]

        if infected_idx.size > 0 and susceptible_idx.size > 0:
            inf_pos = pos[infected_idx]
            sus_pos = pos[susceptible_idx]

            diff = sus_pos[:, None, :] - inf_pos[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)

            close = dist2 <= r2
            for i_sus, near_any in enumerate(close):
                if np.any(near_any) and rng.random() < beta:
                    state[susceptible_idx[i_sus]] = 1

        for i in infected_idx:
            if rng.random() < gamma:
                state[i] = 2

        record(k)

    return S_hist, I_hist, R_hist, pos, stat
