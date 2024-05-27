"""

"""
import rospy
import time
import os
import cv2 
import threading

from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from D_gripperCtrl import *
from std_msgs.msg import Float32MultiArray
from PID import PID

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    
class DGripper_ros(DgripperCtrl):
    '''
    DGripper的ROS控制接口, 三种控制模式
    0 - 订阅5个目标电机位置直接来控制电机
    1 - 订阅位姿变换信息与接触点信息, 计算后控制电机
    2 - 订阅两个触觉图像与位姿变换信息, 处理后控制电机
    3 - rod 控制
    9 - 程序停止模式
    通过publisher不断发送电机位置信息
    '''
    def __init__(self, servo_port, servo_baud, board_port, board_baud):
        super(DGripper_ros, self).__init__(servo_port, servo_baud, board_port, board_baud)
        self.CTRL_MODE = 0
        # ROS
        rospy.init_node("DGripper_ros")
        # 五个电机位置信息 N20[0\1] 位移, N20[2]夹爪间距, servo[0\1] 角度 
        self.pub = rospy.Publisher("/motors_pos_real", Float32MultiArray, queue_size=10)
        # 左右两个摄像头的图像存储列表
        self.BUF_SIZE = 10
        self.image_left_list = []
        self.image_right_list = []
        self.image_sub_left  = rospy.Subscriber("/image_left", Image, self.image_left_callback)
        self.image_sub_right = rospy.Subscriber("/image_right", Image, self.image_right_callback)
        # 电机位置命令订阅  -  数据格式同pub_msg
        self.motor_sub = rospy.Subscriber("/motor_pos_tar", Float32MultiArray, self.motor_callback)
        # 位姿变换命令订阅 
        self.pose_sub = rospy.Subscriber("/pose_tar", Pose, self.pose_callback)
        # 轴类物体旋转任务命令
        self.rod_sub = rospy.Subscriber("/rod_roll", Float32MultiArray, self.rod_roll_callback)
        # self.rod_roll_pid = PID(1.0, 0.01, 0.0, ROLL_MAX_SPD, 0.0, 0.0)     # 旋转pid, 传送带基速度
        self.rod_move_pid = PID(0.02, 0.01, 0.0, ROLL_MAX_SPD/6, 0.0, 0.0)   # 移动pid, 平移修正速度
        self.rod_open_pid = PID(0.2, 0.01, 0.0, SCREW_MAX_SPD, 0.0, 0.0)    # 张合pid, 夹爪张合速度  
        
    def image_left_callback(self, msg):
        '''
        图像订阅回调函数
        '''
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord('y'):
            #保存最新数据
            self.save_latest_data()
        if key == ord('q'):
            quit()
        # 将图像数据添加到缓存列表中
        self.image_left_list.append(image)
        # 如果缓存列表超过了最大数目，则删除最旧的数据
        if len(self.image_left_list) > self.BUF_SIZE:
            self.image_left_list.pop(0)

    def image_right_callback(self, msg):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord('y'):
            #保存最新数据
            self.save_latest_data()
        if key == ord('q'):
            quit()
        # 将图像数据添加到缓存列表中
        self.image_right_list.append(image)
        # 如果缓存列表超过了最大数目，则删除最旧的数据
        if len(self.image_right_list) > self.BUF_SIZE:
            self.image_right_list.pop(0)

    def motor_callback(self, msg):
        '''
        电机位置命令订阅回调函数
        '''
        if self.CTRL_MODE == 9:
            return
        self.CTRL_MODE = 0
        self.state_tar = list(msg.data)
        tar_pos = [self.state_tar[0]/N20_RAD2DIS, self.state_tar[1]/N20_RAD2DIS, (self.state_tar[2]-SCREW_INIT_DIS)/SCREW_RAD2DIS]
        for i in range(3):
            self.board.motor[i].PID_ctrl.set_target(tar_pos[i])
        print("target: ", self.state_tar)

    def pose_callback(self, msg):
        '''
        位姿变换命令订阅回调函数
        '''
        if self.CTRL_MODE == 9:
            return
        self.CTRL_MODE = 1
        # self.pose_tar = msg
        print("pose callback: ", self.pose_tar)
    
    def rod_roll_callback(self, msg):
        '''
        轴类物体旋转任务
        msg: 自旋速度, 触觉图像形心(z,x) *2, 触觉图像深度积分值*2 
        '''
        if self.CTRL_MODE == 9:
            return
        self.CTRL_MODE = 3
        TAR_Z = 0 # 触觉图像上目标z坐标
        TAR_DEPTH = 110 # 触觉图像上目标深度积分值
        K_SPD0 = 0.8
        K_SPD1 = 0.8
        
        tar_roll_spd = msg.data[0]
        center = [  [msg.data[1]+7, msg.data[2]],
                    [msg.data[3]+7, msg.data[4]] ]
        deep = [msg.data[5], msg.data[6]]
        print("msg: ", msg.data)
        self.rod_move_pid.SetPoint = TAR_Z
        move_fix_spd = self.rod_move_pid.cal_output(center[0][0]+ center[1][0])
        self.state_tar[5] = -K_SPD0*(tar_roll_spd + move_fix_spd)
        self.state_tar[6] =  K_SPD1*(tar_roll_spd - move_fix_spd)
        self.rod_open_pid.SetPoint = TAR_DEPTH
        open_spd = self.rod_open_pid.cal_output(deep[0] + deep[1])*1.15
        self.state_tar[7] = open_spd
        print("tar_roll_spd: ", self.state_tar[5], self.state_tar[6], "open_spd: ", self.state_tar[7])
    
    def run(self):
        # Create a rate object with a rate of 10Hz
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # rospy.spinOnce()
            if self.CTRL_MODE == 0:
                self.excute_motor(0)
            elif self.CTRL_MODE == 3:
                self.excute_motor(1)
            elif self.CTRL_MODE == 9:
                self.reset_zero()
                exit(0)
            
            self.update()
            # 电机位置信息发布
            # pub_msg_empty = Float32MultiArray(data = [] )
            # self.pub.publish(pub_msg_empty)
            pub_msg = Float32MultiArray(data = self.state)
            self.pub.publish(pub_msg)
            rate.sleep()
    

# 定义一个函数来检测按键
def check_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            # b 键用于停止
            if sys.stdin.read(1) == 'b':
                print("stop by keypress")
                gripper.CTRL_MODE = 9
                time.sleep(1)
                pid = os.getpid()
                os.system("kill -9 " + str(pid))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


SERVO_BAUDRATE              = 1000000            
SERVO_PORTNAME              = '/dev/ttyUSB0'
BOARD_BAUDRATE              = 115200    
BOARD_PORTNAME              = '/dev/ttyUSB1' 
if __name__ == '__main__':
    # 创建并启动按键检测线程
    keypress_thread = threading.Thread(target=check_keypress)
    keypress_thread.daemon = True  # 设置为守护线程，当主程序退出时，线程也会退出
    keypress_thread.start()
    gripper = DGripper_ros(SERVO_PORTNAME, SERVO_BAUDRATE, BOARD_PORTNAME, BOARD_BAUDRATE)
    gripper.run()
