"""

"""

import time
from STM32Ctrl import STM32Ctrl
import sys
import os
import signal

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

sys.path.append("..")
from scservo_sdk import *                      # Uses SCServo SDK library

PI = 3.1415926536
N20_RAD2DIS = 1.95                        # translate angle(rad) of n20 to displacement(mm) of surface

SCREW_RAD2DIS = 1.35                         # translate screw's angle(rad) to surface distance(mm) of 2 fingers
SERVO_VAL2ANG = 0.0015339807878                # translate servo's angle digital value(0~4095) to actual value(+-PI)
# range limitation of state
ROLL_MAX_SPD = 3
SCREW_MAX_DIS = 35
SCREW_MAX_SPD = 4
SCREW_MIN_DIS = 0
SERVO_MAX_ANG =  PI
SERVO_MIN_ANG = -PI
SCREW_INIT_DIS = 27.2                           # motor pos=0 <--> dis=122.8mm, assume that the thickness of sensor is 45mm, init_dis = 122.8-45x2=32.8mm  
SERVO_INIT_VAL_0 = 2010
SERVO_INIT_VAL_1 = 2095

def limit(val, min_val, max_val):
    if val > max_val:
        val = max_val
    elif val < min_val:
        val = min_val
    return val

# calculate the distance between two state
def cal_dis(state1, state2):
    dis = 0
    for i in range(3):
        dis += (state1[i] - state2[i]) ** 2
    dis = dis ** 0.5
    return dis

class DgripperCtrl:
    SERVO_NUM = 2
    def __init__(self, servo_port, servo_baud, board_port, board_baud):
        ## STM32
        self.board = STM32Ctrl(board_port, board_baud)
        
        ## Feetech servo
        self.servo_pos = [0] * self.SERVO_NUM
        self.servo_delta_pos = 60                       # the delta position after pressing one time
        self.portHandler = PortHandler(servo_port)
        self.packetHandler = sms_sts(self.portHandler)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        # Set port baudrate
        if self.portHandler.setBaudRate(servo_baud):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()
        
        self.servo_pos[0] = self.read_servo(0)
        self.servo_pos[1] = self.read_servo(1)
        # state: surface displacement(mm) of 2 sensors & distance(mm) of 2 fingers & 2 servo's target angle(rad); 
        self.state = [0, 0, 0, 0, 0]
        # add 3 N20s' speed (rad/s)
        self.state_tar = [0, 0, 30, 0, 0, 0, 0, 0]

    def write_servo(self, servo_id, position, speed, acc):
        scs_comm_result, scs_error = self.packetHandler.WritePosEx(servo_id, position, speed, acc)
        # if scs_comm_result != COMM_SUCCESS:
        #     print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        # elif scs_error != 0:
        #     print("%s" % self.packetHandler.getRxPacketError(scs_error))

    def read_servo(self, servo_id):
        scs_present_position, scs_present_speed, scs_comm_result, scs_error = self.packetHandler.ReadPosSpeed(servo_id)
        # if scs_comm_result != COMM_SUCCESS:
        #     print(self.packetHandler.getTxRxResult(scs_comm_result))
        # elif scs_error != 0:
        #     print(self.packetHandler.getRxPacketError(scs_error))
        moving, scs_comm_result, scs_error = self.packetHandler.ReadMoving(servo_id)
        # if scs_comm_result != COMM_SUCCESS:
        #     print(self.packetHandler.getTxRxResult(scs_comm_result))
        # else:
        #     print("[ID:%03d] PresPos:%d PresSpd:%d" % (servo_id, scs_present_position, scs_present_speed))
        # if scs_error != 0:
        #     print(self.packetHandler.getRxPacketError(scs_error))
        return scs_present_position

    def key_run(self):
        while True:
            print("Press any key to continue! (or press ESC to quit!)")
            key = getch()
            # if key = contrl + c
            #
            if key == chr(27):     # ESC
                print("Quit!")
                # pid = os.getpid()
                # os.kill(pid, signal.SIGKILL)
                os._exit(0)
                
            elif key == "q":        # servo 0 clockwise
                self.write_servo(0, self.servo_pos[0] + self.servo_delta_pos, 500, 50)
                print("servo 0 clockwise")
            elif key == "a":        # servo 0 counter-clockwise
                self.write_servo(0, self.servo_pos[0] - self.servo_delta_pos, 500, 50)
                print("servo 0 counter-clockwise")
            elif key == "w":        # servo 1 clockwise
                self.write_servo(1, self.servo_pos[1] + self.servo_delta_pos, 500, 50)
                print("servo 1 clockwise")
            elif key == "s":        # servo 1 counter-clockwise
                self.write_servo(1, self.servo_pos[1] - self.servo_delta_pos, 500, 50)
                print("servo 1 counter-clockwise")
            elif key == "u":        # n20[0] clockwise
                self.board.motor[0].set_speed(3, self.board.ser)
                print("n20[0] clockwise")
            elif key == "j":        # n20[0] counter-clockwise
                self.board.motor[0].set_speed(-3, self.board.ser)
                print("n20[0] counter-clockwise")
            elif key == "i":        # n20[1] clockwise
                self.board.motor[1].set_speed(2.2, self.board.ser)
                print("n20[1] clockwise")
            elif key == "k":        # n20[1] counter-clockwise
                self.board.motor[1].set_speed(-2.2, self.board.ser)
                print("n20[1] counter-clockwise")
            elif key == "o":        # n20[2] clockwise, open
                self.board.motor[2].set_speed(10, self.board.ser) 
                print("n20[2] clockwise")
            elif key == "l":        # n20[2] counter-clockwise
                self.board.motor[2].set_speed(-10, self.board.ser)
                print("n20[2] counter-clockwise")
            elif key == "1":        # set position of motor
                motor_id = int(input("motor id:"))
                position_tar = input("position:")
                self.board.motor[motor_id].set_position(position_tar, self.board.ser)
                print("set position of motor", motor_id, "to", position_tar)
            elif key == "2":        # read position of motor
                motor_id = int(input("motor id:"))
                self.board.motor[motor_id].read_position(self.board.ser)
                print("position of motor", motor_id, "is", self.board.motor[motor_id].position)

            elif key == "3":        # read speed of motor
                motor_id = int(input("motor id:"))
                self.board.motor[motor_id].read_speed(self.board.ser)
            elif key == "b":        # back to zero
                self.reset_zero()
            elif key == "p":        # set n20 pos via upper pid 
                motor_id = int(input("motor id:"))
                position_tar = float(input("position:"))
                self.board.motor[motor_id].PID_ctrl.set_target(position_tar)
                while abs(self.board.motor[motor_id].position - position_tar) > 0.1:
                    self.board.motor[motor_id].pidLoop(self.board.ser)
                    time.sleep(0.1)
                self.board.motor[motor_id].stop(self.board.ser)
                print("set position of motor", motor_id, "to", position_tar)

            elif key == " ":        # stop all
                for i in range(3):
                    self.board.motor[i].stop(self.board.ser)
                # self.write_servo(0, 0, 0, 50)
                # self.write_servo(1, 0, 0, 50)
                print("stop all")

            time.sleep(0.05)
            self.servo_pos[0] = self.read_servo(0)
            self.servo_pos[1] = self.read_servo(1)
            self.update()

    def update(self):
        for i in range(3):
            self.board.motor[i].read_position(self.board.ser)
        self.servo_pos[0] = self.read_servo(0)
        self.servo_pos[1] = self.read_servo(1)
        
        self.state[0] = self.board.motor[0].position * N20_RAD2DIS /1.12 /6.21
        self.state[1] = self.board.motor[1].position * N20_RAD2DIS *1.67*0.65*1.28/6.21
        tempx =self.board.motor[2].position * SCREW_RAD2DIS + SCREW_INIT_DIS
        self.state[2] = (-self.board.motor[2].position * SCREW_RAD2DIS + SCREW_INIT_DIS+11.4)*1.875-59.6+41.6-4.6
        self.state[3] = self.servo_pos[0] * SERVO_VAL2ANG - PI
        self.state[4] = self.servo_pos[1] * SERVO_VAL2ANG - PI
        # print("state:", self.state)

    def excute_motor(self,n20_exe_mode=0):
        # limit range()
        self.state_tar[2] = limit(self.state_tar[2], SCREW_MIN_DIS, SCREW_MAX_DIS)
        self.state_tar[3] = limit(self.state_tar[3], SERVO_MIN_ANG, SERVO_MAX_ANG)
        self.state_tar[4] = limit(self.state_tar[4], SERVO_MIN_ANG, SERVO_MAX_ANG)
        self.state_tar[5] = limit(self.state_tar[5], -ROLL_MAX_SPD, ROLL_MAX_SPD)
        self.state_tar[6] = limit(self.state_tar[6], -ROLL_MAX_SPD, ROLL_MAX_SPD)
        self.state_tar[7] = limit(self.state_tar[7], -SCREW_MAX_SPD, SCREW_MAX_SPD)
        # if self.state[2] <= SCREW_MIN_DIS:  # prevent screw from clash into each other, set speed < 0
        #     self.state_tar[7] = limit(self.state_tar[7], -SCREW_MAX_SPD, 0)
        # print("state_tar:", self.state_tar)

        # translate state to target position, excute  stm32 position control
        tar_pos = [self.state_tar[0]/N20_RAD2DIS, self.state_tar[1]/N20_RAD2DIS, (self.state_tar[2]-SCREW_INIT_DIS)/SCREW_RAD2DIS]
        # for i in range(3):
        #     self.board.motor[i].set_position(tar_pos[i], self.board.ser)
        
        # print("mode", n20_exe_mode, "tar_state[7]", self.state_tar[7])
        if n20_exe_mode == 0:            # pc position control
            for i in range(3):
                self.board.motor[i].pidLoop(self.board.ser)  
        elif n20_exe_mode == 1:          # stm32 velcity control
            for i in range(3):
                self.board.motor[i].set_speed(self.state_tar[i+5], self.board.ser)
            
        servo_pos = [self.state_tar[3]/SERVO_VAL2ANG + SERVO_INIT_VAL_0, self.state_tar[4]/SERVO_VAL2ANG + SERVO_INIT_VAL_1]
        self.write_servo(0, int(servo_pos[0]), 500, 50)
        self.write_servo(1, int(servo_pos[1]), 500, 50)
        

    def reset_zero(self):
        '''
        reset all motors to zero position
        In order to ensure the accuracy of position control,
        gripper should be reset to zero position as long as it's powered off.
        '''
        self.state_tar = [0, 0, 35, 0, 0, 0, 0, 0]
        self.state_tar[3] = 0
        self.state_tar[4] = 0
        tar_pos = [self.state_tar[0]/N20_RAD2DIS, self.state_tar[1]/N20_RAD2DIS, (self.state_tar[2]-SCREW_INIT_DIS)/SCREW_RAD2DIS]
        for i in range(3):
            self.board.motor[i].PID_ctrl.set_target(tar_pos[i])
        last_time = time.time()
        while cal_dis(self.state, self.state_tar) > 0.5:
            print(self.state)
            self.excute_motor(1)
            self.update()
            time.sleep(0.1)
            if time.time() - last_time > 1:
                print("reset zero completed!")
                break
    
    def manipulate(self, pose, tar_pose):
        complete = False
        state = 0
        while not complete:
            
            self.update()
            time.sleep(0.05)
            
SERVO_BAUDRATE              = 1000000            
SERVO_PORTNAME              = '/dev/ttyUSB0'
BOARD_BAUDRATE              = 115200    
BOARD_PORTNAME              = '/dev/ttyUSB1' 

if __name__ == '__main__':
    no1 = DgripperCtrl(SERVO_PORTNAME, SERVO_BAUDRATE, BOARD_PORTNAME, BOARD_BAUDRATE)
    no1.key_run()