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


