import numpy as np
from random import randint
from random import uniform
import matplotlib.pyplot as plt

# up, down, left, right
r = np.matrix([[-1, 0, -1, 0],
               [-1, 0, 0, 0],
               [-1, 0, 0, 0],
               [-1, 0, 0, -1],
               [0, 0, -1, 0],
               [-1, -1, -1, -1],
               [0, 0, 0, 0],
               [0, 0, 0, -1],
               [0, -1, -1, 0],
               [0, -1, 0, 0],
               [0, -1, 0, 0],
               [0, -1, 0, -1]])

# print(r)
V = np.matrix(np.zeros([12, 4]))
#Q = np.matrix(np.zeros([12, 4]))
Q = np.random.rand(12, 4)
# print(Q)
for x in range(12):
    for y in range(4):
        if r[x, y] == -1:
            Q[x, y] = -1
# num of states
n_states = 12

# Goal state index
goal_state = 5

# Greediness parameter
epsilon = 0.5

# Discounting parameter(learning parameter)
gamma = 0.9

# probability of going in intended direction
probability = 0.70

# num of episodes
episodes = 1000

prevQ = 0.0
curQ = 0.0
Qs = []

def generate_resultant_state(state, action):
    if action == 0:
        return state - 4
    elif action == 1:
        return state + 4
    elif action == 2:
        return state - 1
    elif action == 3:
        return state + 1


# list of legal moves/actions available in this state
def generate_legal_actions(state):
    current_rewards = r[state, :]
    legal_moves = np.where(current_rewards >= 0)[1]
    return legal_moves


# choosing random action out of available actions
def choose_random_action(available_actions):
    next_action = np.random.choice(available_actions)
    return next_action

# choosing max value generating action - Greedy approach
def choose_max_value_action(current_state, available_actions):
    next_best_action = available_actions[0]
    max_val = 0
    for action in available_actions:
        rsa = r[current_state, action]
        next_state = generate_resultant_state(current_state, action)
        val = Q[next_state, :].max()
        if rsa + val > max_val:
            next_best_action = action
    return next_best_action

def update_Q(current_state, next_state, action, legal_actions):
    rsa = r[current_state, action]
    estimated_rsa = rsa / (V[current_state, action])
    max_val = 0
    max_val += probability*(Q[next_state, :].max())
    for action in legal_actions:
        next_probable_state = generate_resultant_state(current_state, action)
        if next_probable_state == next_state:
            continue
        else:
            max_val += (((1.0 - probability) / (len(legal_actions) - 1)))*(Q[next_probable_state, :].max())
    new_q = estimated_rsa + (gamma * max_val)
    alpha = 1.0 / (1.0 + V[current_state, action])
    Q[current_state, action] = (1.0 - alpha)*Q[current_state, action] + alpha*new_q

def plot(Qs):
    iterations = [x for x in range(episodes)]
    plt.plot(iterations, Qs)
    plt.show()

def evaluate_reward(current_state, next_state, legal_actions):
    num_actions = len(legal_actions)
    if next_state == goal_state:
        return probability*100
    else:
        for action in legal_actions:
            next_probable_state = generate_resultant_state(current_state, action)
            if next_probable_state == goal_state:
                return ((1.0 - probability) / (num_actions - 1))*100
        return 0

for episode in range(episodes):
    current_state = randint(0, n_states - 1)
    while current_state != goal_state:
        legal_actions = generate_legal_actions(current_state)
        random_legal_action = choose_random_action(legal_actions)
        if uniform(0, 1) > (1 - epsilon):
            random_legal_action = choose_random_action(legal_actions)
        else:
            random_legal_action = choose_max_value_action(current_state, legal_actions)
        next_state = generate_resultant_state(current_state, random_legal_action)
        #print(current_state, " " , next_state)
        reward = evaluate_reward(current_state, next_state, legal_actions)
        r[current_state, random_legal_action] += reward
        V[current_state, random_legal_action] += 1
        update_Q(current_state, next_state, random_legal_action, legal_actions)
        current_state = next_state
    for x in range(12):
        for y in range(4):
            curQ += Q[x, y]
    Qs.append(curQ - prevQ)
    prevQ = curQ
    curQ = 0

print(Q)
plot(Qs)
print(r)
print(V)
