import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
"""
-State (s): The agent's current environment.
Usage: Input for the neural network to estimate Q-values.

- Action (a): The action taken by the agent.
Usage: Selects the Q-value to be updated.

- Reward (r): The benefit or penalty received after an action.
Usage: Used to compute the prediction error and target Q-value.

- Next State (s'): The new state after an action.
Usage: Helps determine the future Q-value and update targets.

"""

import numpy as np
import torch


class PrioritizedExpeReplayBuffer(object):


    def __init__(self, capacity, batch_size, PER_alpha, PER_beta, beta_increment_per_sampling, PER_epsilon):
        self.buffer_capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.surprises = []
        self.beta = PER_beta
        self.alpha = PER_alpha
        self.epsilon = PER_epsilon
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def add_experience(self, state, action, reward, Q_target, Q_value, next_state):

        Q_targets_tensor = torch.tensor(Q_target, requires_grad=True).float()
        Q_targets_tensor = Q_targets_tensor.expand_as(Q_value)
        Q_target = Q_target.item()  # Estrae il valore scalare dal tensore
        Q_value = Q_value.item()    # Estrae il valore scalare dal tensore

        surprise = Q_value - (1 - reward)
        experience = (state, action, reward, Q_target, Q_value, next_state)
        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)
            self.surprises.pop(0)
        self.buffer.append(experience)
        self.surprises.append(surprise)

    def sample_batch(self):
        # Assumiamo che self.surprises contenga i valori di sorpresa per ogni esperienza nella memoria
        surprises = np.array(self.surprises)
        
        # Normalizza i valori di sorpresa per ottenere probabilità
        max_surprise = np.max(surprises)
        normalized_surprises = surprises / max_surprise
        probabilities = normalized_surprises / np.sum(normalized_surprises)
        
        # Seleziona gli indici basati sulle probabilità
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]

        states = []
        actions = []
        rewards = []
        next_states = []
        Q_targets = []
        Q_values = []
        for experience in batch:
            state, action, reward, Q_target, Q_value, next_state = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            Q_targets.append(Q_target)
            Q_values.append(Q_value)

        return states, actions, rewards, Q_targets, Q_values, next_states
   
    def get_buffer_length(self):
        return len(self.buffer)

"""
    def __init__(self, capacity, batch_size, PER_alpha, PER_beta, beta_increment_per_sampling, PER_epsilon):
        self.buffer_capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.priorities = []
        self.beta = PER_beta
        self.alpha = PER_alpha
        self.epsilon = PER_epsilon
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def add_experience(self, state, action, reward, Q_target, Q_value, next_state):


        Q_targets_tensor = torch.tensor(Q_target, requires_grad=True).float()
        Q_targets_tensor = Q_targets_tensor.expand_as(Q_value)
        Q_target = Q_target.item()  # Estrae il valore scalare dal tensore
        Q_value = Q_value.item()    # Estrae il valore scalare dal tensore



        error = Q_target - Q_value
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        experience = (state, action, reward, Q_target, Q_value, next_state)
        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample_batch(self):
        priorities = np.array(self.priorities)
        probabilities = priorities / sum(priorities)
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]

        states = []
        actions = []
        rewards = []
        next_states = []
        Q_targets = []
        Q_values = []
        for experience in batch:
            state, action, reward, Q_target, Q_value, next_state = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            Q_targets.append(Q_target)
            Q_values.append(Q_value)
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # Compute importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        return states, actions, rewards, Q_targets, Q_values, next_states, weights
   
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (np.abs(error) + self.epsilon) ** self.alpha

    def get_buffer_length(self):
        return len(self.buffer)


"""
