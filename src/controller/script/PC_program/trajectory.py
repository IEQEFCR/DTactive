import numpy as np
from math import sin,cos, pi
import matplotlib.pyplot as plt

class trajectory:
    time_step = 0.5
    step_range = 600

    def __init__(self):
        self.t = np.arange(0, self.time_step*self.step_range, self.time_step)
        self.desired_pos = np.zeros(self.step_range)
        for i in range(self.step_range):
            self.desired_pos[i] = 200*sin(pi*i/self.step_range)*sin(6*pi*i/self.step_range)

    def draw_trajectory(self):
        # 绘制期望轨迹,添加网格
        plt.plot(self.t, self.desired_pos)
        #y轴细分
        plt.yticks(np.arange(-200, 200, 20))
        plt.grid()
        plt.show()

    def pid_control(self,t,now_pos,P):
        d_pos = self.desired_pos[int(t/trajectory.time_step)]
        error = d_pos - now_pos
        return P*error, d_pos


if __name__ == '__main__':
    t = trajectory()
    t.draw_trajectory()