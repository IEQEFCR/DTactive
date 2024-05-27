#!/bin/env/python3

import rospy
import os

#录制rosbag，录制话题/height_map,/gripper_state
def record():
    rospy.init_node('record', anonymous=True)
    rospy.loginfo("Start record")