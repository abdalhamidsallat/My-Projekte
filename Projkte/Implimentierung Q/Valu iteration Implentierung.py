import numpy as np


def Value_Iteration(R, P, gamma, THETA):
    n, _ = R.shape
    V = np.zeros(shape=n)
    # das ist unserer Ziel V=[0,0,0,0,0,0,1,0,0,0]
    V[6] = 1
    converged = False
    # Rebeat (wiederholen)
    while not converged:
        # ∆ ← 0
        DELTA = 0
        # For each s ∈ S:
        for j in range(len(V)):
            if j == 6:
                continue
            # Verwenden von V_k zur Berechnung von V_(k+1)
            Vold = V.copy()
            # Belohnungen von Nachfolgezustande
            rewards = R[j]
            # Wahrscheinlichkeiten der Nachfolgerzustande
            prob = P[j]
            max_reward = np.max(rewards)
            # Aktionen mit maximalen Belohnungen auswahlen
            next_state = np.where(max_reward == rewards)
            # Wenn next_state mehr als eins ist, dann ist der Durchschnitt über alle diese
            if len(next_state[0]) > 1:
                state = 0
                for n in next_state[0]:
                    state = prob[n] * (max_reward + gamma * Vold[n])
                    V[j] += state
                V[j] /= len(next_state[0])
            # sonst V berechnen
            else:
                V[j] = prob[next_state[0]] * (max_reward + (gamma * Vold[next_state[0]]))
            Vdefernce = Vold - V
            # ∆ ← max(∆, |v − V (S)|)
            DELTA = max(DELTA, np.linalg.norm(Vdefernce))
            # until ∆ < θ(a klein positive zahl)
            converged = True if DELTA < THETA else False

    return V


# Anzal die Platze
garten = 10

R = np.zeros(shape=[garten, garten])
R[0, :] = [-1, 0, -1, 0, -1, -1, -1, 0, 0, -1]
R[1, :] = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
R[2, :] = [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1]
R[3, :] = [0, -1, 0, -1, -1, 0, -1, -1, -1, -1]
R[4, :] = [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1]
R[5, :] = [-1, -1, -1, 0, 0, -1, 1, -1, -1, -1]
R[6, :] = [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1]
R[7, :] = [0, -1, -1, -1, -1, -1, 1, -1, -1, -1]
R[8, :] = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
R[9, :] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1]

P = np.zeros(shape=[garten, garten])
P[0, :] = [0, 0.25, 0, 0, 0, 0, 0, 0.25, 0.25, 0]
P[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
P[2, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
P[3, :] = [0.3, 0, 0.3, 0, 0, 0.4, 0, 0, 0, 0]
P[4, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
P[5, :] = [0, 0, 0, 0, 0.2, 0.3, 0.5, 0, 0, 0]
P[6, :] = [0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0]
P[7, :] = [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0]
P[8, :] = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]
P[9, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

if __name__ == '__main__':
    # a ist beliebege wert fur θ = THETA
    a = 1e-6
V = Value_Iteration(R=R, P=P, gamma=0.9, THETA=a)
print("unser result von V")
print(V)
