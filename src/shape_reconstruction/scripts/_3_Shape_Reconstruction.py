#!/bin/python3

import os
import cv2
import numpy as np
import yaml
from sensor import Sensor
from visualizer import Visualizer
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from matplotlib import pyplot as plt
import open3d as o3d

def tactile_img(sensor,index=1):
    img = sensor.get_rectify_crop_image()
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_GRAY)
    mask[img_GRAY>180] = 255
    mask[img_GRAY<80] = 255
    # mask self.mask或运算
    mask = mask | sensor.mask

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    img_GRAY = cv2.inpaint(img_GRAY, mask, 7, cv2.INPAINT_TELEA)

    # cv2.imshow('RawImage_GRAY', img_GRAY)
    height_map = sensor.raw_image_2_height_map(img_GRAY)
    if index == 2:
        img = cv2.flip(img, 1)
        height_map = cv2.flip(height_map, 1)
    depth_map = sensor.height_map_2_depth_map(height_map)

    #cvtColor depth_map to BGR
    img = cv2.cvtColor(depth_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #除去depth_map中0值以外的排名25%的值
    if np.max(depth_map) > 0:
        rk25 = np.percentile(depth_map[depth_map>0], 20)
    else :
        rk25 = 0
    main_body = depth_map.copy()
    main_body[depth_map < rk25] = 0

    depth_sum = np.sum(main_body)/10000
    depth_sum = int(depth_sum)
    dx, dz = 0, 0
    #findContours in main_body
    contours, _ = cv2.findContours(main_body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #find the max contour
    # main_body = cv2.cvtColor(main_body.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) <700:
                img = cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)
        max_contour = max(contours, key=cv2.contourArea)
        #if the area of the max contour is too small, ignore it
        if cv2.contourArea(max_contour) > 800:
            #draw the max contour
            # cv2.drawContours(img, [max_contour], -1, 255, -1)
            #mask_contour的重心
            M = cv2.moments(max_contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            sensor.center_point.append((cx, cy))
            dx = img.shape[0]//2-cy
            dz = cx - img.shape[1]//2
            if len(sensor.center_point) > 5:
                sensor.center_point.pop(0)
            avg_center = np.mean(sensor.center_point, axis=0)

            # cv2.circle(img, (int(avg_center[0]), int(avg_center[1])), 5, (0, 255, 0), -1)

            #轴线提取
            #轮廓的最小外接矩形
            rect = cv2.minAreaRect(max_contour)
            #矩形的四个顶点
            box = cv2.boxPoints(rect)
            #矩形的长轴，画出
            dis1 = np.linalg.norm(box[0]-box[1])
            dis2 = np.linalg.norm(box[1]-box[2])
            if dis1 < dis2:
                mid1 = (box[0]+box[1])/2
                mid2 = (box[2]+box[3])/2
            else:
                mid1 = (box[1]+box[2])/2
                mid2 = (box[3]+box[0])/2
            sensor.axis_queue.append((mid1, mid2))
            if len(sensor.axis_queue) > 5:
                sensor.axis_queue.pop(0)
            avg_axis = np.mean(sensor.axis_queue, axis=0)
            # cv2.line(img, (int(avg_axis[0][0]), int(avg_axis[0][1])), (int(avg_axis[1][0]), int(avg_axis[1][1])), (0, 255, 0), 2)

        #霍夫圆检测
        canny = cv2.Canny(img, 50, 150)

        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                # 输出像素直径
                cv2.putText(img, 'd: '+str(i[2]*2), (i[0]+10, i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(img, 'r: '+str(i[2]), (i[0]+10, i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(img, 'x: '+str(i[0]), (i[0]+10, i[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(img, 'y: '+str(i[1]), (i[0]+10, i[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    # cv2.putText(img, 'depth_sum: '+str(depth_sum), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, 'dx: '+str(dx), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, 'dz: '+str(dz), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img,dx,dz,depth_sum,height_map

def heightmap2point(heightmap):
    w,h = heightmap.shape
    point = heightmap[w//2,:]*16
    point.reshape((h,))
    # show = np.zeros((w,h,3), np.uint8)
    # peak=[]
    # for i in range(h):
    #     show[int(point[i]*16),i] = [255,255,255]
    #     if(i>5 and i+5<h and point[i] > point[i-5] and point[i] > point[i+5]):
    #         show[int(point[i]*16),i] = [0,0,255]
    #         peak.append(i)
    # if (len(peak)>1):print(peak[-1]-peak[0])
    #point 加上一列 range(0, h)
    #去除等于0的点
    point = np.vstack((point, np.arange(0,h)))
    point = np.vstack((point, np.zeros(h)))
    point = point[:,point[0,:]>0]
    h = len(point)

    if(h>0):
        return point
    else:
        return np.zeros((3,460))

if __name__ == '__main__':
    rospy.init_node('tactile_node', anonymous=True)
    rod_pub = rospy.Publisher("/rod_roll", Float32MultiArray, queue_size=10)
    key_pub = rospy.Publisher('/teleop', String, queue_size=10)
    height_map_pub = rospy.Publisher("/height_map", Image, queue_size=10)
    depth_map_pub = rospy.Publisher("/tactile_image", Image, queue_size=10)
    depth_max_pub = rospy.Publisher("/depth_max", Float32MultiArray, queue_size=10)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    f = open("shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['camera_setting']['camera_channel'] =0
    sensor1 = Sensor(cfg)
    cfg['camera_setting']['camera_channel'] =2
    sensor2 = Sensor(cfg)

    index = 0
    # visualizer = Visualizer(sensor.points)
    # depth_map_video = cv2.VideoWriter('depth_map.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (460,345))
    video_path ='./video/'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    while os.path.exists(video_path+str(index)+'_demo.mp4'):
        index += 1
    video = cv2.VideoWriter(video_path+str(index)+'_demo.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 25, (460,690))     
    start_video = False
    bridge = CvBridge()
    
    rate = rospy.Rate(20)

    saved_trigger = False
    last_time = rospy.Time.now()
    avg_img = np.zeros((690,460,3), np.float32)
    cnt = 0
    t_axis= []
    max_axis = []
    time_start = rospy.Time.now()

    vis = o3d.visualization.Visualizer()
    # vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((460,3)))
    for i in range(460):
        pcd.points[i] = [i,0,0]
    # vis.add_geometry(pcd)

    while sensor1.cap.isOpened() and sensor2.cap.isOpened():
        t1, dx1, dz1, depth_sum1 ,h1= tactile_img(sensor1,1)
        t2 ,dx2, dz2, depth_sum2 ,h2= tactile_img(sensor2,2)

        # point1 = heightmap2point(h2)
        # point2 = heightmap2point(h1)
        # point1[0] = -300 -point1[0]
        # point = np.hstack((point1, point2))
        # pcd.points = o3d.utility.Vector3dVector(point1.T)

        msg_data = [2, dz1, dx1, dz2, dx2, depth_sum1, depth_sum2]
        pub_msg = Float32MultiArray(data=msg_data)
        rod_pub.publish(pub_msg)

        show_img = np.vstack((t1, t2))
        height_map = np.vstack((h1, h2))

        height_map_pub.publish(bridge.cv2_to_imgmsg(height_map, "32FC1"))
        depth_map_pub.publish(bridge.cv2_to_imgmsg(show_img, "bgr8"))
        # print max height
        print("\033c")
        depth_max_pub.publish(Float32MultiArray(data=[np.max(h1)+np.max(h2)]))

        cv2.line(show_img, (0,345), (460,345), (255,255,255), 2)
        cv2.putText(show_img, 'Left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(show_img, 'Right', (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Tactile', show_img)

        if start_video:
            video.write(show_img)
        key = cv2.waitKey(1)

        # if key == ord('s'):
        #     if start_video:
        #         print('Stop recording')
        #         video.release()
        #         index += 1
        #         video = cv2.VideoWriter(video_path+str(index)+'_demo.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 25, (460,690))
        #     else:
        #         print('Start recording')
        #     start_video = not start_video

        if key == ord('q'):
            msg = String()
            #esc键的ascii码为27
            msg.data = chr(27)
            # print(msg.data)
            key_pub.publish(msg)
            break

        if saved_trigger:
            avg_img += show_img
            cnt += 1
            if cnt == 50:
                print('Save image')
                cv2.imwrite('avg_img.png', (avg_img/cnt).astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                print('Time cost: ', rospy.Time.now()-last_time)
                saved_trigger = False
        
        if key == ord('c'):
            if not saved_trigger:
                saved_trigger = True
                avg_img  = np.zeros_like(avg_img)
                avg_img += show_img
                cnt = 1
                last_time = rospy.Time.now()
            # 无损保存图片

        if key != -1:
            #translate key to ascii
            key = chr(key)
            msg = String()
            msg.data = str(key)
            # print(msg.data)
            key_pub.publish(msg)

        rate.sleep()
