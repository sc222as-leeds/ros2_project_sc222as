# Exercise 1 - Display an image of the camera feed to the screen

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from math import sin, cos, atan2
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('navigation_goal_action_client')
        self.fov = 1.085595
        self.sensitivity = 10
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        self.bridge = CvBridge()
        self.blue_size = 0
        self.blue_threshold = 202959.5

        self.position = [0.0, 0.0]
        self.orientation = 0.0
        self.goal_handle = None

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)

    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # NOTE: if you want, you can use the feedback while the robot is moving.
        #       uncomment to suit your need.

        # Access the current pose
        current_pose = feedback_msg.feedback.current_pose
        self.position = [current_pose.pose.position.x, current_pose.pose.position.y]
        q = current_pose.pose.orientation
        self.orientation = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def search(self):
        current_idx = 0
        coordinates = [[-9.7, 2.6, 0.0], [-8.95, -4.77, 0.0], [-8.23, -13.6, 0.0], [0.0, -13.0, 0.0], [7.85, -12.2, 0.0], [7.65, -4.4, 0.0], [6.79, 4.64, 0.0], [-3.57, 3.8, 0.0], [-1.49, -4.31, 0.0]]
        coord = coordinates[current_idx]
        pos_goal = [coord[0], coord[1]]

        self.send_goal(coord[0], coord[1], coord[2])
        while current_idx != len(coordinates):
            if abs(self.position[0] - coord[0]) < 0.2 and abs(self.position[1] - coord[1]) < 0.2 and abs(self.orientation - coord[2]) < 0.3:
                current_idx += 1
                coord = coordinates[current_idx]
                pos_goal = [coord[0], coord[1]]
                self.turn_360()
                if self.blue_in_front:
                    return
                
                self.send_goal(coord[0], coord[1], coord[2])

    def turn_360(self):
        twist = Twist()

        twist.angular.z = 0.2
        speed = abs(twist.angular.z)

        accumulated_rotation = 0.0

        # Start time
        start_time = time.time()

        while accumulated_rotation < (2 * np.pi):
            self.cmd_vel_pub.publish(twist)
            current_time = time.time()
            change_time = current_time - start_time
            start_time = current_time

            if self.blue_in_front:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                
                self.move_towards_blue()
                return

            accumulated_rotation += speed * change_time
            self.rate.sleep()

        twist.angular.z = 0.0
        twist.angular.x = 0.0
        self.cmd_vel_pub.publish(twist)
        print("Stopped", flush=True)

    def move_towards_blue(self):
        twist = Twist()
        velocity = 0.2
        angle_per = self.fov / self.image_width
        twist.linear.x = velocity
        
        while self.blue_size < self.blue_threshold:
            # print("moving to blue")
            offset = self.blue_center - self.image_center
            yaw_offset = angle_per * offset # Multiply offset with scaling factor
            
            # current_yaw = self.orientation
            # target_yaw = current_yaw - yaw_offset
            # target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi

            twist.angular.z = -yaw_offset * 0.2
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()
        
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_center = image.shape[1] // 2
            self.image_width = image.shape[1]
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            hsv_green_lower = np.array([60 - self.sensitivity, 50, 50])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
            
            hsv_blue_lower = np.array([120 - self.sensitivity, 50, 50])
            hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
            
            hsv_red_lower1 = np.array([0, 100, 100])
            hsv_red_lower2 = np.array([180 - self.sensitivity, 50, 50])
            hsv_red_upper1 = np.array([0 + self.sensitivity, 255, 255])
            hsv_red_upper2 = np.array([180, 255, 255])
            
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
            red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
            red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
            
            red_mask = red_mask1 | red_mask2
            # print(np.count_nonzero(green_mask))
            
            rgb_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(blue_mask, red_mask))
            
            filtered_img = cv2.bitwise_and(image, image, mask=rgb_mask)
            
            contours, _ = cv2.findContours(blue_mask,mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(biggest_contour)
                
                M = cv2.moments(biggest_contour)
                if M["m00"] != 0:
                    self.blue_in_front = True
                    self.blue_size = area
                    self.blue_center = int(M["m10"] / M["m00"])
                else:
                    self.blue_in_front = False
            else:
                self.blue_in_front = False
            
            cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
            cv2.imshow('camera_Feed', filtered_img)
            cv2.resizeWindow('camera_Feed', 320, 240)
            cv2.waitKey(1)
        except:
            pass
        

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():

    def signal_handler(sig, frame):
        rclpy.shutdown()
    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    cI.search()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()
    

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
