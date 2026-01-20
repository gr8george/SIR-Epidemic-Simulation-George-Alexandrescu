import numpy as np

def euler_sir(beta, gamma, S0, I0, R0, dt, steps):
    """
    Euler method for the SIR model.
    """
    t = np.arange(steps + 1) * dt
    S = np.zeros(steps + 1)
    I = np.zeros(steps + 1)
    R = np.zeros(steps + 1)

    S[0], I[0], R[0] = S0, I0, R0

    for k in range(steps):
        dS = -beta * S[k] * I[k]
        dI = beta * S[k] * I[k] - gamma * I[k]
        dR = gamma * I[k]

        S[k + 1] = S[k] + dt * dS
        I[k + 1] = I[k] + dt * dI
        R[k + 1] = R[k] + dt * dR

        total = S[k + 1] + I[k + 1] + R[k + 1]
        if total > 0:
            S[k + 1] /= total
            I[k + 1] /= total
            R[k + 1] /= total

    return t, S, I, R

