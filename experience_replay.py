import random
import numpy as np
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


class ExperienceReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.buffer = []
        self.buffer_capacity = capacity
        self.batch_size = batch_size

    #modo 1 per aggiungere esperienza
    #def add_experience(self, experience):
    #    if len(self.buffer) >= self.buffer_capacity:
    #        self.buffer.pop(0)          #if buffer full, then remove the oldest experience --> pop(0) rimuove elemento piu vecchio, gli altri scalano
    #    self.buffer.append(experience)

    # Add a new experience to the buffer
    def add_experience(self, state, action, reward, next_state):
        
        # Create a new experience
        experience = (state, action, reward, next_state)

        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)                          #if buffer full, then remove the oldest experience --> pop(0) rimuove elemento piu vecchio, gli altri scalano
        
        # Add the new experience
        self.buffer.append(experience)

    # Extract a (random) batch from the buffer
    def sample_batch(self):
        if self.get_buffer_length() <= self.batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, self.batch_size)

        return batch
    
    #modo 2
    #def sample_batch(self, batch_size):
    #    batch = random.sample(self.buffer, batch_size)
    #    states, actions, rewards, next_states = zip(*batch)
    #    return states, actions, rewards, next_states
    
    # Give buffer lenght (number of episodes in the buffer)
    def get_buffer_length(self):
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)