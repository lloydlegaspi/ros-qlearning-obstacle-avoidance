#!/usr/bin/env python3
import rospy
import os
import time
from environment import Env
from agent import DQNAgent

def main():
    # Initialize the ROS Node
    rospy.init_node('turtlebot3_dqn_trainer')
    
    # Define Hyperparameters for the Training Loop
    EPISODES = 300
    MAX_STEPS = 500
    
    # Initialize Environment and Agent
    env = Env()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Create a directory to save the trained model weights
    save_dir = os.path.dirname(os.path.realpath(__file__)) + '/save_model/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rospy.loginfo("Starting DQN Training for TurtleBot3...")
    start_time = time.time()

    for e in range(EPISODES):
        # Check if user pressed Ctrl+C
        if rospy.is_shutdown():
            break
            
        done = False
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            if rospy.is_shutdown():
                break
                
            # 1. Choose an action
            action = agent.get_action(state)
            
            # 2. Execute action in Gazebo
            next_state, reward, done = env.step(action)
            
            # 3. Save experience to memory
            agent.append_sample(state, action, reward, next_state, done)
            
            # 4. Train the neural network
            agent.train_model()
            
            total_reward += reward
            state = next_state
            
            # 5. Handle Episode End
            if done:
                # Update target network at the end of each episode
                agent.update_target_model()
                break
                
        # --- DATA LOGGING FOR YOUR REPORT ---
        # This exact format makes it easy to parse into a CSV later
        print(f"EPISODE: {e+1}, REWARD: {total_reward:.2f}, EPSILON: {agent.epsilon:.4f}", flush=True)
        
        # Save the model every 10 episodes
        if (e + 1) % 10 == 0:
            agent.update_target_model()
            # Saving weights (Note: requires torch to be imported in agent.py)
            import torch
            torch.save(agent.model.state_dict(), save_dir + f"turtlebot3_dqn_{e+1}.pt")
            
    # Calculate total training time
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    rospy.loginfo(f"Training Completed in {h}h {m}m {s}s")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass