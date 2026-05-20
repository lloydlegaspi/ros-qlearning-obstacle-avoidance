#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

class Env:
    def __init__(self):
        # Set up publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.get_scan)
        
        # Gazebo services for resetting the simulation
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        
        # State variables
        self.scan_data = None
        self.collision_distance = 0.20 # Robot crashes if obstacle is closer than 20cm
        self.action_size = 3 # 0: Left, 1: Straight, 2: Right
        self.state_size = 5  # 5 buckets of LiDAR data

    def get_scan(self, msg):
        self.scan_data = msg.ranges

    def discretize_scan(self):
        # Wait until LiDAR data is available
        while self.scan_data is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        # Replace infinite values with max range
        scan = np.array(self.scan_data)
        scan = np.where(np.isinf(scan), 3.5, scan) 
        
        # Concatenate the front-left and front-right slices to form a single front cone
        front_cone = np.concatenate((scan[0:36], scan[324:360]))
        
        # The Turtlebot LiDAR has 360 points. We bucket them into 5 sectors for the AI.
        state = [
            np.min(front_cone),               # Front (narrow cone crossing 0 degrees)
            np.min(scan[36:108]),             # Left-Front
            np.min(scan[108:180]),            # Left
            np.min(scan[180:252]),            # Right
            np.min(scan[252:324])             # Right-Front
        ]
        return np.array(state)

    def step(self, action):
        # 1. Execute the action
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15 # Constant forward speed
        
        if action == 0:
            vel_cmd.angular.z = 1.0   # Turn Left
        elif action == 1:
            vel_cmd.angular.z = 0.0   # Go Straight
        elif action == 2:
            vel_cmd.angular.z = -1.0  # Turn Right
            
        self.pub_cmd_vel.publish(vel_cmd)
        
        # Allow the simulator to process the movement
        rospy.sleep(0.1) 
        
        # 2. Get the new state
        next_state = self.discretize_scan()
        
        # 3. Calculate Reward and Check if Done (Collision)
        done = False
        min_distance = np.min(next_state)
        
        if min_distance < self.collision_distance:
            done = True
            reward = -50.0 # Heavy penalty for crashing
        else:
            # Minor reward for surviving, heavily scaled by distance to nearest wall
            # This encourages the robot to stay in the middle of hallways
            reward = 1.0 + (min_distance - 0.5) 
            
        return next_state, reward, done

    def reset(self):
        # Stop the robot
        vel_cmd = Twist()
        self.pub_cmd_vel.publish(vel_cmd)
        
        # Call Gazebo's reset service
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            
        # Return the initial state
        rospy.sleep(0.5) # Give Gazebo a moment to settle
        return self.discretize_scan()