// include ros header
#include <ros/package.h>
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Float32MultiArray.h>


using namespace std;

class angle_tracking {
   public:
    int windows_size = 5, index =0; 
    double ground_truth_angle = 0, last_yaw = 480,sliding_windows[20];
    bool get_new_aruco = false,reset = false,first_aruco = true;
    ros::Publisher angle_pub;
    ros::Subscriber aruco_sub,teleop_sub;

    void aruco_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        get_new_aruco = true;
        // aruco_pose = *msg;
        double roll, pitch, yaw;
        tf::Quaternion quat;
        tf::quaternionMsgToTF(msg->pose.orientation, quat);
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        yaw = yaw * 180 / 3.1415926;
        double delta_yaw = yaw - last_yaw;
        if(first_aruco||fabs(yaw - last_yaw) > 90)
            delta_yaw = 0,first_aruco=false;
        ground_truth_angle += delta_yaw;
        last_yaw = yaw;
    }

    void teleop_callback(const std_msgs::String::ConstPtr& msg) {
       if(msg->data == "r")
           ground_truth_angle = 0,first_aruco = true;
    }

    angle_tracking() {
        ros::NodeHandle nh;
        aruco_sub = nh.subscribe("/aruco_single/pose", 1,
                                 &angle_tracking::aruco_callback, this);
        angle_pub = nh.advertise<std_msgs::Float32MultiArray>("/angle", 1);
        teleop_sub = nh.subscribe("/teleop", 1, &angle_tracking::teleop_callback, this);

        ros::Rate loop_rate(25);

        while (ros::ok()) {
            if (get_new_aruco) {
                std_msgs::Float32MultiArray angle_msg;
                index++;
                if(index>windows_size) index=1;
                sliding_windows[index] = ground_truth_angle;
                double avg = 0;
                for(int i=1;i<=windows_size;i++)
                    avg +=sliding_windows[i];
                avg/=windows_size;

                angle_msg.data.push_back(avg);
                angle_pub.publish(angle_msg);
                get_new_aruco = false;
                ROS_INFO("angle: %f", avg);
            }
            ros::spinOnce();
            loop_rate.sleep();
        }

    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "tactile_pose_tracking");
    angle_tracking at;
    return 0;
}
