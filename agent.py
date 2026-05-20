#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# The Neural Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 5 inputs -> 64 hidden nodes -> 64 hidden nodes -> 3 output actions
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters (Perfect data points to analyze in your report)
        self.discount_factor = 0.99   # Gamma
        self.learning_rate = 0.001    # Alpha
        self.epsilon = 1.0            # Initial exploration rate
        self.epsilon_decay = 0.995    # How fast it transitions to exploitation
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 1000       # Start training after 1000 steps of random exploration
        
        # Replay Memory
        self.memory = deque(maxlen=10000)
        
        # Initialize Networks
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model() # Sync weights initially
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        # Epsilon-Greedy Strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values[0]).item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        # Decay Epsilon gradually
        if self.epsilon > self.epsilon_min and len(self.memory) > self.train_start:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
            
        # Sample a random batch from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in mini_batch]))
        actions = torch.LongTensor(np.array([x[1] for x in mini_batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([x[2] for x in mini_batch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in mini_batch]))
        
        # Current Q values
        q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))
            
        # Optimize
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()