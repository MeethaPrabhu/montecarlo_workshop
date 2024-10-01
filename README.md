# montecarlo_workshop

## AIM:
To predict the optimal state-value function for an agent navigating a Frozen Lake environment using the Monte Carlo Prediction method. 

## Problem Statement:
In the Frozen Lake environment, the objective is to predict the state-value function using the Monte Carlo Prediction method, where an agent navigates the environment under a random policy. The goal is to estimate the expected future rewards for each state based on simulated episodes, helping evaluate the agent's performance in reaching the goal while avoiding holes.

## Monte Carlo Algorithm:
### Step 1:
Initialize state-value function V(s) for all states.
### Step 2:
Generate episodes by following a given policy.
### Step 3:
For each state in an episode, calculate the return G and update V(s).
### Step 4:
Repeat for n_episodes until convergence.
### Step 5:
Return the final estimated state-value function.

## Implementation of decay_schedule:
```
import numpy as np
def decay_schedule(init_value, min_value,decay_ratio, max_steps, log_start=-2, log_base=10):
  decay_steps = int(max_steps * decay_ratio)
  rem_steps = max_steps - decay_steps
#This function allows you to calculate all the values for alpha for the full training process.
#(2) First, calculate the number of steps to decay the values using the decay_ratio variable. (3) Then, calculate the actual values as an inverse log curve. Notice we then normalize between 0 and 1, and finally transform the points to lay between init_value and min_value.
  values = np.logspace (log_start, 0, decay_steps,base=log_base, endpoint=True) [::-1]
  values =(values - values.min()) / (values.max()-values.min())
  values = (init_value - min_value) * values + min_value
  values = np.pad(values, (0, rem_steps), 'edge')
  return values
```

## Implementation of generate_trajectory:
```
from itertools import count

def generate_trajectory(pi, env, max_steps=20):
  done, trajectory = False, []
  while not done:
    state = env.reset()
    for t in count():
      action=pi(state)
      next_state, reward, done,_=env.step(action)
      experience = (state,action, reward, next_state, done)
      trajectory.append(experience)
      if done:
        break
      if t>=max_steps - 1:
        trajectory = []
        break
      state=next_state
    return np.array(trajectory, dtype=object)
```

## Implementation of  mc_prediction:
```
from tqdm import tqdm
def mc_prediction(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3, n_episodes=500, max_steps=100, first_visit=True):
    ns = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)  # Fixed the 'endpoint' argument
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    v = np.zeros(ns)
    v_track = np.zeros((n_episodes, ns))
    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(pi, env, max_steps)
        visited = np.zeros(ns, dtype=bool)
        for t, (state, action, reward, next_state, done) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            v[state] = v[state] + alphas[e] * (G - v[state])
        v_track[e] = v
    return v.copy(), v_track
```

## Printing Functions:
```
# Print policy function
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```

## Printing State Value Function:
```
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```

## Monte Carlo Prediction:
```
# Printing the policy (Monte Carlo doesn't compute an optimal policy directly, so we'll use random policy for now)
print("Name: Meetha Prabhu      Register Number: 212222240065")
print('State-value function from Monte Carlo prediction:')

# Get the environment's transition dynamics
P = env.env.P

# Print the random policy (used for Monte Carlo) and the state-value function from Monte Carlo prediction
print_policy(random_policy, P)
print_state_value_function(V_mc, P, prec=4)

# Calculate and print the success rate and the mean return for the Monte Carlo results
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, random_policy, goal_state=goal_state)*100,
    mean_return(env, random_policy)))
```
### Output:
![image](https://github.com/user-attachments/assets/ae78fdb7-f2b0-46c8-91e7-7e62678a464e)





