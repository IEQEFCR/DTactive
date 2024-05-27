import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

class Data:
    def __init__(self, data_path):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path ='./dataset/'
        self.data_path = self.dataset_path + data_path
        self.txt_path = self.data_path + '/data.txt'
        self.real = []
        self.esti = []
        self.img = []
        self.read_data()
    def read_data(self):
        data = []
        with open(self.txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                self.real.append(float(line[2]))
                self.esti.append(float(line[4]))
        for i in range(len(self.real)):
            img_path = self.data_path + '/' + str(i) + '.jpg'
            img = cv.imread(img_path)
            self.img.append(img)
        self.img = self.img[3:380]
        self.real = self.real[3:380]
        self.esti = self.esti[3:380]
    def show_data(self):
        plt.plot(self.real, label='real')
        plt.plot(self.esti, label='esti')
        error = np.array(self.real) - np.array(self.esti)
        plt.plot(error, label='error')
        max_error = max(abs(error))
        #put text
        plt.text(0, max_error, 'max error: '+str(max_error))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    #change worch dir to this file's dir


    data = Data('5')
    data.show_data()