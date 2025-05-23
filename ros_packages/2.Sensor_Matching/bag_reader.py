""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class BagReader(Node):
    def __init__(self, bag_path):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/depth/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

def depth_to_xyz(depth_image, depth_intrin):
    points = []
    for y_i in range(depth_intrin.height):
        for x_i in range(depth_intrin.width):
            
            depth = depth_image[y_i][x_i] * 0.001  # Convert from mm to meters
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_i, y_i], depth)
                points.append(point)
            elif depth == 0:            #############
                point = [0, 0, 0]       ##########
                points.append(point)    #######
    return np.array(points)

def main(args=None):
    rclpy.init(args=args)





    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3')
    bag_reader.read_bag()
    
    if bag_reader.depth_image is not None and bag_reader.color_image is not None and bag_reader.depth_intrin is not None:
        print(bag_reader.depth_image)
        points = depth_to_xyz(bag_reader.depth_image, bag_reader.depth_intrin)
        print(points)
        print(np.array(points).shape)
    rclpy.shutdown()

if __name__ == '__main__':
    main() """



""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2

class BagReader(Node):
    def __init__(self, bag_path):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/aligned_depth_to_color/image_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/aligned_depth_to_color/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Visualize the frames
            if self.depth_image is not None and self.color_image is not None:
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Stack both images horizontally
                images = np.hstack((self.color_image, depth_colormap))

                # Show images
                cv2.imshow('Color and Depth Frames', images)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

def depth_to_xyz(depth_image, depth_intrin):
    points = []
    for y in range(depth_intrin.height):
        for x in range(depth_intrin.width):
            depth = depth_image[y, x] * 0.001  # Convert from mm to meters
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                points.append(point)
    return np.array(points)

def main(args=None):
    rclpy.init(args=args)
    bag_reader = BagReader()
    bag_reader.read_bag()

    if bag_reader.depth_image is not None and bag_reader.color_image is not None and bag_reader.depth_intrin is not None:
        points = depth_to_xyz(bag_reader.depth_image, bag_reader.depth_intrin)
        print(points)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
 """


""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/aligned_depth_to_color/image_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/aligned_depth_to_color/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Visualize the frames
            if self.depth_image is not None and self.color_image is not None:
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Stack both images horizontally
                images = np.hstack((self.color_image, depth_colormap))

                # Show images
                cv2.imshow('Color and Depth Frames', images)
                if cv2.waitKey(self.playback_speed) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

def depth_to_xyz(depth_image, depth_intrin):
    points = []
    for y in range(depth_intrin.height):
        for x in range(depth_intrin.width):
            depth = depth_image[y, x] * 0.001  # Convert from mm to meters
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                points.append(point)
    return np.array(points)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 20  # Adjust this value to control playback speed (milliseconds)
    
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/2023_10_04_rosbag-piante/bag5/bag5_0.db3', playback_speed)
    bag_reader.read_bag()

    if bag_reader.depth_image is not None and bag_reader.color_image is not None and bag_reader.depth_intrin is not None:
        points = depth_to_xyz(bag_reader.depth_image, bag_reader.depth_intrin)
        print(points)

    rclpy.shutdown()

if __name__ == '__main__':
    main()

 """

""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/aligned_depth_to_color/image_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/aligned_depth_to_color/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Visualize the frames
            if self.depth_image is not None and self.color_image is not None:
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Stack both images horizontally
                images = np.hstack((self.color_image, depth_colormap))

                # Show images
                cv2.imshow('Color and Depth Frames', images)
                key = cv2.waitKey(self.playback_speed) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused

                while self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        self.paused = not self.paused
                    elif key == ord('q'):
                        break

                # Publish the point cloud
                points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                self.publisher.publish(point_cloud_msg)

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_data = []

        for i in range(len(points)):
            point = points[i]
            color = color_image[i // color_image.shape[1], i % color_image.shape[1]]
            rgb = struct.unpack('I', struct.pack('BBB', color[2], color[1], color[0]))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 30  # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/2023_10_04_rosbag-piante/bag5/bag5_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main() """


""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            print(topic)
            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if topic == '/camera/depth/camera_info':
                print('OK')
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Check if the depth_intrin is correctly set before proceeding
            if self.depth_intrin is not None:
                # Visualize the frames
                if self.depth_image is not None and self.color_image is not None:
                    # Apply colormap to depth image for visualization
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    
                    # Stack both images horizontally
                    images = np.hstack((self.color_image, depth_colormap))

                    # Show images
                    cv2.imshow('Color and Depth Frames', images)
                    key = cv2.waitKey(self.playback_speed) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused

                    while self.paused:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('p'):
                            self.paused = not self.paused
                        elif key == ord('q'):
                            break

                    # Publish the point cloud
                    points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                    point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                    self.publisher.publish(point_cloud_msg)
            else:
                self.get_logger().warn('Depth intrinsic parameters not set. Skipping frame.')

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_cloud_data = []

        for i in range(len(points)):
            point = points[i]
            y = i // color_image.shape[1]
            x = i % color_image.shape[1]
            color = color_image[y, x]
            rgb = struct.unpack('I', struct.pack('BBB', color[2], color[1], color[0]))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 50  # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main() """


""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/depth/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Check if the depth_intrin is correctly set before proceeding
            if self.depth_intrin is not None:
                # Visualize the frames
                if self.depth_image is not None and self.color_image is not None:
                    # Apply colormap to depth image for visualization
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    
                    # Stack both images horizontally
                    images = np.hstack((self.color_image, depth_colormap))

                    # Show images
                    cv2.imshow('Color and Depth Frames', images)
                    key = cv2.waitKey(self.playback_speed) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused

                    while self.paused:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('p'):
                            self.paused = not self.paused
                        elif key == ord('q'):
                            break

                    # Publish the point cloud
                    
                    points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                    point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                    self.publisher.publish(point_cloud_msg)
                    self.get_logger().info('Publishing...')
            else:
                self.get_logger().warn('Depth intrinsic parameters not set. Skipping frame.')

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
                    
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_cloud_data = []

        for i in range(len(points)):
            point = points[i]
            y = i // color_image.shape[1]
            x = i % color_image.shape[1]
            color = color_image[y, x]
            rgb = struct.unpack('I', struct.pack('I', (color[2] << 16) | (color[1] << 8) | color[0]))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 50  # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
 """
""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/depth/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Check if both images and depth_intrin are correctly set before proceeding
            if self.depth_image is not None and self.color_image is not None and self.depth_intrin is not None:
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                images = np.hstack((self.color_image, depth_colormap))

                # Show images
                cv2.imshow('Color and Depth Frames', images)
                key = cv2.waitKey(self.playback_speed) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused

                while self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        self.paused = not self.paused
                    elif key == ord('q'):
                        break

                # Publish the point cloud
                points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                self.publisher.publish(point_cloud_msg)
            else:
                self.get_logger().warn('Depth or color image or intrinsic parameters not set. Skipping frame.')

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_cloud_data = []

        for i in range(len(points)):
            point = points[i]
            y = i // color_image.shape[1]
            x = i % color_image.shape[1]
            color = color_image[y, x]
            rgb = struct.unpack('I', struct.pack('I', (color[2] << 16) | (color[1] << 8) | color[0]))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 30  # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main() """

    
""" import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/depth/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            # Check if both images and depth_intrin are correctly set before proceeding
            if self.depth_image is not None and self.color_image is not None and self.depth_intrin is not None:
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                images = np.hstack((self.color_image, depth_colormap))

                # Show images
                cv2.imshow('Color and Depth Frames', images)
                key = cv2.waitKey(self.playback_speed) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused

                while self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        self.paused = not self.paused
                    elif key == ord('q'):
                        break

                # Publish the point cloud
                points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                self.publisher.publish(point_cloud_msg)
            else:
                self.get_logger().warn('Depth or color image or intrinsic parameters not set. Skipping frame.')

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_cloud_data = []

        height, width, _ = color_image.shape
        for i in range(len(points)):
            point = points[i]
            x = int((point[0] / point[2]) * self.depth_intrin.fx + self.depth_intrin.ppx)
            y = int((point[1] / point[2]) * self.depth_intrin.fy + self.depth_intrin.ppy)
            if 0 <= x < width and 0 <= y < height:
                color = color_image[y, x]
                rgb = (color[2] << 16) | (color[1] << 8) | color[0]
                point_cloud_data.append([point[0], point[1], point[2], struct.unpack('f', struct.pack('I', rgb))[0]])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 30  # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
 """

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import pyrealsense2 as rs
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import struct

class BagReader(Node):
    def __init__(self, bag_path, playback_speed):
        super().__init__('bag_reader')
        self.bridge = CvBridge()
        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.depth_image = None
        self.color_image = None
        self.depth_intrin = None
        self.paused = False
        self.publisher = self.create_publisher(PointCloud2, 'colored_pointcloud', 10)

    def read_bag(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        i = 0
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            if topic == '/camera/depth/image_rect_raw':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif topic == '/camera/color/image_raw':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == '/camera/depth/camera_info':
                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = msg.width
                self.depth_intrin.height = msg.height
                self.depth_intrin.ppx = msg.k[2]
                self.depth_intrin.ppy = msg.k[5]
                self.depth_intrin.fx = msg.k[0]
                self.depth_intrin.fy = msg.k[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = msg.d

            i +=1
            if i % 20 == 0:
                # Check if both images and depth_intrin are correctly set before proceeding
                if self.depth_image is not None and self.color_image is not None and self.depth_intrin is not None:
                    # Apply colormap to depth image for visualization
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

                    # Stack both images horizontally
                    images = np.hstack((self.color_image, depth_colormap))

                    # Show images
                    cv2.imshow('Color and Depth Frames', images)
                    key = cv2.waitKey(self.playback_speed) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused

                    while self.paused:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('p'):
                            self.paused = not self.paused
                        elif key == ord('q'):
                            break

                    # Publish the point cloud
                    
                    points = self.depth_to_xyz(self.depth_image, self.depth_intrin)
                    point_cloud_msg = self.create_colored_pointcloud(points, self.color_image)
                    self.publisher.publish(point_cloud_msg)
                else:
                    self.get_logger().warn('Depth or color image or intrinsic parameters not set. Skipping frame.')

        cv2.destroyAllWindows()

    def depth_to_xyz(self, depth_image, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                depth = depth_image[y, x] * 0.001  # Convert from mm to meters
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_cloud_data = []

        height, width, _ = color_image.shape
        for i in range(len(points)):
            point = points[i]
            x = int((point[0] / point[2]) * self.depth_intrin.fx + self.depth_intrin.ppx)
            y = int((point[1] / point[2]) * self.depth_intrin.fy + self.depth_intrin.ppy)
            if 0 <= x < width and 0 <= y < height:
                color = color_image[y, x]
                rgb = (color[2] << 16) | (color[1] << 8) | color[0]  # RGB format
                # Encode the RGB value as UINT32
                rgb_uint32 = struct.unpack('I', struct.pack('BBBB', color[0], color[1],color[2], 0))[0]
                point_cloud_data.append([point[0], point[1], point[2], rgb_uint32])

        return create_cloud(header, fields, point_cloud_data)

def main(args=None):
    rclpy.init(args=args)
    playback_speed = 100      # Adjust this value to control playback speed (milliseconds)
    bag_reader = BagReader('/home/tandaniele/Documents/Thesis/2.Sensor_Matching/bag_all_topic/rosbag2_2024_06_04-11_18_15_0.db3', playback_speed)
    bag_reader.read_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
