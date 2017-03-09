import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import math

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
estimated_r = np.matrix(np.zeros([12, 4]))
Q = np.matrix(np.zeros([12, 4]))
# print(Q)
for x in range(12):
    for y in range(4):
        if r[x, y] == -1:
            Q[x, y] = -1
# num of states
n_states = 12

# Goal state index
goal_state = 5

# Discounting parameter(learning parameter)
gamma = 0.9

# Greediness parameter
epsilon = 1.0

# probability of going in intended direction
probability = 0.70

# num of episodes
episodes = 10000

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
    legal_moves = np.squeeze(np.asarray(legal_moves))
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
        temp_val = Q[current_state, action]
        if temp_val > max_val:
            next_best_action = action
            max_val = temp_val
    return next_best_action

# choosing random action out of available actions
def update_Q(legal_actions):
    reward = 0.0
    available_actions = legal_actions
    final_action = legal_actions[0]
    if random.uniform(0, 1) > (1 - epsilon):
        next_action = choose_random_action(legal_actions)
    else:
        next_action = choose_max_value_action(current_state, available_actions)
    remaining_actions = np.delete(available_actions, np.argwhere(available_actions == next_action))
    rand = random.uniform(0, 1)
    if rand <= 0.7:
        final_action = next_action
    else:
        num = rand - 0.7
        den = (1.0 - 0.7) / (remaining_actions.size)
        final_action = remaining_actions[num / den]
    next_state = generate_resultant_state(current_state, final_action)
    if next_state == goal_state:
        reward = 100.0
    r[current_state, next_action] += reward
    V[current_state, next_action] += 1
    estimated_r[current_state, next_action] = (r[current_state, next_action] / V[current_state, next_action])
    new_q = reward + (gamma * (Q[next_state, :].max()))
    alpha = 1.0 / (1.0 + V[current_state, next_action])
    Q[current_state, next_action] = (1.0 - alpha)*Q[current_state, next_action] + alpha*new_q
    return next_state

def plot(Qs):
    iterations = [x for x in range(episodes)]
    plt.plot(iterations, Qs)
    plt.show()

for episode in range(episodes):
    current_state = randint(0, n_states - 1)
    while current_state != goal_state:
        legal_actions = generate_legal_actions(current_state)
        next_state = update_Q(legal_actions)
        current_state = next_state
    for x in range(12):
        for y in range(4):
            curQ += Q[x, y]
    Qs.append(math.fabs(curQ - prevQ))
    prevQ = curQ
    curQ = 0

print(Q)
plot(Qs)
print(estimated_r)