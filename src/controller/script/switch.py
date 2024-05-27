#!/usr/bin/env python3
import os
from time import sleep
#change working directory to the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

program0 = "python3 ./PC_program/0D_gripper_ROS.py"
program1 = "python3 ./PC_program/1D_gripperCtrl.py"

#运行program1,若程序退出，范围值为0，则运行program2
now = 1

while True:
    if now == 1:
        ret = os.system(program1)
    else:
        ret = os.system(program0)
    if ret == 0:
        break
    now = 1 - now 
    sleep(0.3)