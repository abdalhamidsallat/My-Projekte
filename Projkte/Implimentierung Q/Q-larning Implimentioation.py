
import numpy as np

# Funktion
def q_learning(R, gamma, alpha, episodes):
    """funktion fur Q Learning"""
    # Anyahl der Zeilen und Spalten der R- Matrix
    n,p = R.shape

    # Erstellung der Q Matix (0-werte)
    Q = np.zeros(shape=[n,p])

    # Loop Episoden
    for i in range(episodes):
        # Zufalliger Startpunkt des Roboters
        state = np.random.randint(0, n, 1)

        # Iteration
        for j in range(100):

            # Mogliche Rewards (Belohnungen) im Aktuellen Zustand
            rewards = R[state]

            # Mogliche Bewegungen des Roboters im Aktuellen Zustand
            possible_moves = np.where(rewards[0] > -1)[0]

            # Zufallige Bewegung des Roboters
            next_state = np.random.choice(possible_moves,1)

            # Aktualisierung der Q werte berechnen
            Q[state,next_state] = Q[state,next_state]+alpha*(R[state,next_state]+ gamma * np.max(Q[next_state, :])-Q[state,next_state])

            # Abbrechen der Episode wenn Ziel erreicht
            if R[state, next_state] == 1:
                break

    # Q-Matrix zurückgeben
    return Q



# Anzahl der Garten
garten = 10

#reward matrix
R = np.zeros(shape=[garten,garten])
R[0, :] = [-1, 0 , -1,  0, -1, -1, -1,  0,  0, -1]
R[1, :] = [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
R[2, :] = [-1, -1, -1,  0, -1, -1, -1, -1, -1, -1]
R[3, :] = [ 0, -1,  0, -1, -1,  0, -1, -1, -1, -1]
R[4, :] = [-1, -1, -1, -1, -1,  0, -1, -1, -1, -1]
R[5, :] = [-1, -1, -1,  0,  0, -1,  0, -1, -1, -1]
R[6, :] = [-1, -1, -1, -1, -1,  0, -1,  0, -1, -1]
R[7, :] = [ 0, -1, -1, -1, -1, -1,  1, -1, -1, -1]
R[8, :] = [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
R[9, :] = [-1, -1, -1, -1, -1, -1, -1, -1,  0, -1]

# probability matrix
P = np.zeros(shape=[garten,garten])
R[0, :] = [  0 ,0.25,  0 ,  0 ,  0 ,  0 ,  0 ,0.25,0.25,  0 ]
R[1, :] = [  1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ]
R[2, :] = [  0 ,  0 ,  0 ,  1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ]
R[3, :] = [ 0.3,  0 , 0.3,  0 ,  0 , 0.4,  0 ,  0 ,  0 ,  0 ]
R[4, :] = [  0 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ,  0 ,  0 ,  0 ]
R[5, :] = [  0 ,  0 ,  0 ,  0 , 0.3, 0.3, 0.4,  0 ,  0 ,  0 ]
R[6, :] = [  0 ,  0 ,  0 ,  0 ,  0 , 0.5,  0 , 0.5,  0 ,  0 ]
R[7, :] = [ 0.5,  0 ,  0 ,  0 ,  0 ,  0 , 0.5,  0 ,  0 ,  0 ]
R[8, :] = [ 0.5,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 0.5]
R[9, :] = [  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ]

# value iteration
def Value_Iter(R, P, gamma, episodes ):
    n, _ = R.shape
    V = np.zeros(shape = n)
    V[5] = 1 #
    for i in range(episodes):
        for j in range(len(V)):
            if j == 5:
                continue
            # Verwenden von V_k zur Berechnung von V_(k+1)
            Vold = V.copy()

            #Belohnungen von Nachfolgezustande
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
            else:
                V[j] = prob[next_state[0]] * (max_reward + (gamma * Vold[next_state[0]]))
    return V

# Valu iteration
V = Value_Iter(R=R ,P=P, gamma=0.95,episodes=20)
print(V)

# Q-Learning
Q = q_learning(R=R, gamma=0.95, alpha=0.8, episodes=1000)

print(Q)
q = np.round(Q/np.max(Q), 4)
print(q)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('nicee')

