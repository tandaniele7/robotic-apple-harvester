import rclpy
import cv2
import sys
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
import struct
import rclpy.time
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
import pyrealsense2 as rs
from sensor_msgs_py.point_cloud2 import create_cloud



class PCL2(Node):
    def __init__(self, height=720, width=1280):
        super().__init__("pcl_generator")
        rclpy.logging.set_logger_level("ROSCamera", 10)

        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        # depth_topic = "/camera/depth/image_rect_raw"
        
        color_topic = "/camera/color/image_raw"
        
        # camera_info_depth_topic = "/camera/depth/camera_info"
        camera_info_depth_topic = "/camera/aligned_depth_to_color/camera_info"
        
        pcl_topic = "/camera/depth/color/points_manual"
        self.frame_id = "world"
        self.W = width
        self.max_depth = 5.0
        self.H = height


        self.bridge = CvBridge()
        self.image = []
        self.depth_image = []
        self.pc2 = PointCloud2()
        self.depth_intrin = None

        self.pub = self.create_publisher(PointCloud2, pcl_topic, 10)

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile=qos_profile_sensor_data
        )

        self.color_sub = self.create_subscription(
            Image, color_topic, self.color_callback, qos_profile=qos_profile_sensor_data
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_depth_topic,
            self.info_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.spin_callback()
        timer_period = 5
        self.timer = self.create_timer(timer_period, self.spin_callback)
        self.get_logger().info("Initializing process")

    def spin_callback(self):
        # print(np.array(self.depth_image).shape)
        if (
            np.array(self.depth_image).shape != ()
            and np.array(self.depth_image).shape != (0, 0)
            and np.array(self.depth_image).shape != (0,)
            and np.array(self.image).shape != ()
            and np.array(self.image).shape != (0, 0)
            and np.array(self.image).shape != (0,)
            and self.depth_intrin != None
        ):
            points = []
            
            # depth_t = np.array(self.depth_image).transpose()
            depth_t = self.depth_image
            
            points_xyz = self.depth_to_xyz(depth_t, self.depth_intrin)

            self.pc2 = self.create_colored_pointcloud(
                color_image=self.image, points=points_xyz
            )
            self.pub.publish(self.pc2)
            self.get_logger().info("PCL received")
        # rclpy.spin_once(self)

    def depth_callback(self, msg):

        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #print(np.array(self.depth).shape)
        max_depth = self.max_depth  # [m]
        # IF REAL CAMERA
        # max_depth = self.cutoff*1000 #[mm]

        #depth_frame = np.nan_to_num(self.depth, nan=0.0, posinf=max_depth, neginf=0.0)
        """ spatial = rs.spatial_filter()
        depth_frame = rs.video_frame(self.depth)
        depth_frame = spatial.process(depth_frame)
        self.depth_image = np.asanyarray(depth_frame.get_data()) """
        0
        # self.depth_image = cv2.GaussianBlur(self.depth, (5, 5), 0) <-- this makes the cone
        self.depth_image = self.depth

        
                # Resize the depth image
        # self.depth_image = cv2.resize(self.depth_image, (self.W, self.H), interpolation=cv2.INTER_NEAREST)


        """ depth_frame[depth_frame == 0.0] = max_depth
        #print([np.max(depth_frame), np.min(depth_frame)])
        depth_frame = np.minimum(
            depth_frame, max_depth
        )  """  # [m] in simulation, [mm] with real camera

        # if using a pretrained backbone which normalize data, scale in [0.,255.]
        # depth_frame = depth_frame*255.0

        """ depth_frame = depth_frame.astype(dtype=np.float32)
        self.depth = np.array(depth_frame) """
        #self.display(self.depth_image, depth=True, window_name='Depth Gazebo')

    def color_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # self.image = cv2.resize(self.color, (self.H, self.W),interpolation=cv2.INTER_LINEAR)
        self.image = self.color
        
        # self.display(self.color, depth=False, window_name='RGB')

    def info_callback(self, msg):
        self.depth_intrin = rs.intrinsics()
        self.depth_intrin.width = msg.width
        self.W = msg.width
        self.depth_intrin.height = msg.height
        self.H = msg.height
        self.depth_intrin.ppx = msg.k[2]
        self.depth_intrin.ppy = msg.k[5]
        self.depth_intrin.fx = msg.k[0]
        self.depth_intrin.fy = msg.k[4]
        self.depth_intrin.model = rs.distortion.brown_conrady
        self.depth_intrin.coeffs = msg.d

    def display(self, img, depth=False, window_name="Image"):
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

        if not depth:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, img)
                cv2.waitKey(1)
            except:
                print("Error in opening frame. Image shape: {}".format(img.shape))
        else:
            try:
                """ img = np.nan_to_num(img, nan=0.0, posinf=self.max_depth, neginf=0.0)
                img = np.minimum(img, self.max_depth)
                img = img / np.max(img) * 255.0
                colormap = np.asarray(img, dtype=np.uint8) """
                colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow(window_name, colormap)
                cv2.waitKey(1)
            except:
                print("Error in opening frame. Image shape: {}".format(img.shape))

    def depth_to_xyz(self, dpt, depth_intrin):
        points = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                dpth_ = dpt[y, x] * 0.001  # Convert from mm to meters
                if dpth_ > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dpth_)
                    points.append(point)
        return np.array(points)

    def create_colored_pointcloud(self, points, color_image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_data = []

        height, width, _ = color_image.shape
        for i in range(len(points)):
            point = points[i]
            x = int(
                (point[0] / point[2]) * self.depth_intrin.fx + self.depth_intrin.ppx
            )
            y = int(
                (point[1] / point[2]) * self.depth_intrin.fy + self.depth_intrin.ppy
            )
            if 0 <= x < width and 0 <= y < height:
                color = color_image[y, x]
                rgb = (color[2] << 16) | (color[1] << 8) | color[0]  # RGB format
                # Encode the RGB value as UINT32
                rgb_uint32 = struct.unpack(
                    "I", struct.pack("BBBB", color[0], color[1], color[2], 0)
                )[0]
                rgb_float32 = struct.unpack('f', struct.pack('I', rgb_uint32))[0]
                point_cloud_data.append([point[0], point[1], point[2], rgb_float32])

        return create_cloud(header, fields, point_cloud_data)


def main(args=None):
    rclpy.init()
    pic4depth = PCL2()

    try:
        rclpy.spin(pic4depth)
    except KeyboardInterrupt:
        pic4depth.get_logger().info("Shutting down")

    pic4depth.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()