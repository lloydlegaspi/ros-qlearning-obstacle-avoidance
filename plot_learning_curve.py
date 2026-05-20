#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 1. Define file paths
data_file = 'training_data.txt'
output_image = 'learning_curve.png'

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found in the current directory.")
    exit()

episodes = []
rewards = []

# 2. Parse the text file
print(f"Reading data from {data_file}...")
with open(data_file, 'r') as file:
    for line in file:
        # Extract Episode and Reward using Regular Expressions
        match = re.search(r"EPISODE:\s*(\d+),\s*REWARD:\s*([-\d.]+)", line)
        if match:
            episodes.append(int(match.group(1)))
            rewards.append(float(match.group(2)))

if not episodes:
    print("No valid data found to plot. Check your training_data.txt format.")
    exit()

# 3. Calculate a moving average (window of 5 for an 80-episode run)
window = 5
if len(rewards) >= window:
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ma_episodes = episodes[window-1:]
else:
    moving_avg = rewards
    ma_episodes = episodes

# 4. Generate the Plot
plt.figure(figsize=(10, 6))

# Plot raw data as a faded background line
plt.plot(episodes, rewards, alpha=0.4, color='cornflowerblue', label='Raw Episode Reward')

# Plot the smoothed trendline on top
plt.plot(ma_episodes, moving_avg, color='darkblue', linewidth=2.5, label=f'{window}-Episode Moving Average')

# 5. Formatting for a professional report look
plt.title('DQN Agent Learning Curve (80 Episodes)', fontsize=14, fontweight='bold')
plt.xlabel('Training Episode', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Save the image in high resolution for your paper
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Success! Graph saved as {output_image}")