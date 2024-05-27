#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
#导入rqt_plot
import rqt_plot
from rqt_plot.data_plot import DataPlot
from std_msgs.msg import String
import os
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class Estimator:
    def depth_max_callback(self, data):
        self.depth_max = data.data[0]-40.5+29.32

    def gripper_callback(self, data):
        data =data.data

        if(self.is_first):
            for i in range(3):
                self.gripper_state[i] = data[i]
            self.is_first = False
            return 

        for i in range(3):
            if(abs(data[i]-self.gripper_state[i])<3): #防止数据异常
                self.dx[i] = data[i]-self.gripper_state[i]
                self.gripper_state[i] = data[i]
                self.back_store[i] +=self.dx[i] #回程误差存储
                
                if self.back_store[i] > self.back_limit:
                    self.dx[i] = self.back_store[i] - self.back_limit
                    self.back_store[i] = self.back_limit
                elif self.back_store[i] < -self.back_limit:
                    self.dx[i] = self.back_store[i] + self.back_limit
                    self.back_store[i] = -self.back_limit
                else :
                    self.dx[i] = 0
            else : #数据异常时，用之前的数据
                self.dx[i]/=2
                self.gripper_state[i] += self.dx[i]

    def teleop_callback(self, msg):
        self.received_time = rospy.get_time()
        if(msg.data[0]=='r') :
            self.reset= True
            self.switch = True

    def tactile_img_callback(self, data):
        #ros img to cv img

        if(self.switch):
            self.is_recording = not self.is_recording
            self.switch = False
            if(self.is_recording):
                self.now_index = 0
                self.save_path = "/home/kai/dataset/"+str(rospy.get_time())+"/"
                os.makedirs(self.save_path)

        if(self.is_recording):
            img_save_path = self.save_path + str(self.now_index)+".jpg"
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv.imwrite(img_save_path,cv_img)
            write_data = str(self.now_index)+" real: "+str(self.angle)+" esti: "+str(self.theta)+"\n"
            #向文件中写入数据，不断追加
            with open(self.save_path+"data.txt","a") as f:
                f.write(write_data)
            self.now_index += 1
                
    def angle_callback(self, data):
        self.angle = data.data[0]

    def __init__(self):
        rospy.init_node("estimator")
        self.depth_max_sub = rospy.Subscriber("/depth_max", Float32MultiArray, self.depth_max_callback)
        self.gripper_sub = rospy.Subscriber("/motors_pos_real", Float32MultiArray, self.gripper_callback)
        self.angle_sub = rospy.Subscriber("/angle", Float32MultiArray, self.angle_callback)
        self.orientation_pub = rospy.Publisher("/orientation", Float32MultiArray, queue_size=10)
        self.teleop_sub = rospy.Subscriber("/teleop", String, self.teleop_callback)
        self.tactile_img_sub = rospy.Subscriber("/tactile_image", Image, self.tactile_img_callback)
        self.reset = False
        self.depth_max = 0
        self.gripper_state = [0, 0, 0]
        self.dx=[0,0,0]
        self.back_store = [0,0,0]
        self.back_limit = 0.3
        self.is_first = True
        self.switch = False
        self.is_recording = False
        self.save_path = "/home/kai/dataset/"
        self.now_index = 0
        self.write_file = None
        self.angle = 0
        self.theta = 0
        self.bridge = CvBridge()

    def run(self):
        d_fixed = self.depth_max + self.gripper_state[2]
        theta = 0
        avg_dx0=[]
        avg_dx1=[]
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            d_fixed = self.depth_max + self.gripper_state[2]
            temp= (self.dx[1]-self.dx[0])/d_fixed

            if(temp<0.15 and temp>-0.15):
                self.theta -= 1.127*temp*180/3.1415926
            
            if self.reset==True:
                self.theta = 0
                self.is_first = True
                self.reset = False

            avg_dx0.append(self.dx[0])
            avg_dx1.append(-self.dx[1])

            if(len(avg_dx0)>1000): avg_dx0= avg_dx0[-1000:]
            if(len(avg_dx1)>1000): avg_dx1= avg_dx1[-1000:]

            avg_dx0_ = sum(avg_dx0)/len(avg_dx0)
            avg_dx1_ = sum(avg_dx1)/len(avg_dx1)

            self.orientation_pub.publish(Float32MultiArray(data=[self.theta,d_fixed,avg_dx0_,avg_dx1_]))
            #clear screen
            print("\033c")
            print("theta:",theta,"d_fixed:",d_fixed)

            rate.sleep()

if __name__ == "__main__":
    estimator = Estimator()
    rospy.sleep(1)
    estimator.run()
            
            
            
            
