import rclpy
import cv2
import sys
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import struct
import rclpy.time
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
import pyrealsense2 as rs
from sensor_msgs_py.point_cloud2 import create_cloud, read_points
import open3d as o3d
from masks_detection import detect_masks

MODEL_PATH = (
    "/workspaces/vscode_ros2_workspace/1.Training/runs/segment/train3/weights/best.pt"
)

np.set_printoptions(threshold=sys.maxsize)


def float_to_rgb(float_rgb):
    packed = struct.pack("f", float_rgb)
    integers = struct.unpack("I", packed)[0]    
    r = (integers >> 16) & 0xFF
    g = (integers >> 8) & 0xFF
    b = integers & 0xFF
    return r, g, b


class Detection(Node):
    def __init__(self):
        super().__init__("pcl_to_img")
        rclpy.logging.set_logger_level("PCL2imgTransformer", 10)

        camera_info_depth_topic = "/camera/aligned_depth_to_color/camera_info"
        pcl_topic = "/camera/depth/color/points_manual"
        reconstruct_img_topic = "/img/img_regen"
        color_topic = "/camera/color/image_raw"
        apple_mask_topic = "/img/apple_mask"
        apple_pcl_topic = "/pcl/apples"
        branches_pcl_topic = "/pcl/branches"
        metalwire_pcl_topic = "/pcl/metal_wire"
        leaves_pcl_topic = "/pcl/leaves"

        self.pcl2 = None
        self.W = 1208
        self.cam_info = None
        self.H = 720

        self.image_rgb = []
        self.bridge = CvBridge()
        self.depth_intrin = None

        # Create Open3D point cloud object
        self.pcd = o3d.geometry.PointCloud()

        # publishers and subscribers
        self.apple_mask_pub = self.create_publisher(Image, apple_mask_topic, 10)
        self.img_pub = self.create_publisher(Image, reconstruct_img_topic, 10)
        self.apple_pub = self.create_publisher(PointCloud2, apple_pcl_topic, 10)
        self.branches_pub = self.create_publisher(PointCloud2, branches_pcl_topic, 10)
        self.metal_wire_pub = self.create_publisher(
            PointCloud2, metalwire_pcl_topic, 10
        )
        self.leaves_pub = self.create_publisher(PointCloud2, leaves_pcl_topic, 10)

        self.pcl_sub = self.create_subscription(
            PointCloud2,
            pcl_topic,
            self.pcl_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_depth_topic,
            self.cam_info_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.spin_callback()
        self.timer_period = 10

        self.timer = self.create_timer(self.timer_period, self.spin_callback)
        self.get_logger().info("Initializing process")

    def color_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.image_rgb = self.color

    def pcl_callback(self, msg):
        self.points = []
        for point in read_points(
            msg, field_names=("x", "y", "z", "rgb"), skip_nans=False
        ):
            x, y, z, rgb = point
            r, g, b = float_to_rgb(rgb)
            self.points.append([x, y, z, r, g, b])

        points = np.array(self.points)
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        self.pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.0)

    def cam_info_callback(self, msg):
        self.depth_intrin = rs.intrinsics()
        self.depth_intrin.width = msg.width
        self.W = msg.width
        self.depth_intrin.height = msg.height
        self.H = msg.height
        self.depth_intrin.ppx = msg.k[2]
        self.depth_intrin.ppy = msg.k[5]
        self.depth_intrin.fx = msg.k[0]
        self.depth_intrin.fy = msg.k[4]
        self.cam_info_k = msg.k
        self.depth_intrin.model = rs.distortion.brown_conrady
        self.depth_intrin.coeffs = msg.d

    def spin_callback(self):

        # Project point cloud to 2D image using CameraInfo
        width = self.W
        height = self.H
        apple_points = []
        branches_points = []
        wire_points = []
        leaves_points = []

        # Create an empty image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if (np.array(self.image_rgb).shape != ()
            and np.array(self.image_rgb).shape != (0, 0)
            and np.array(self.image_rgb).shape != (0,)
            and self.depth_intrin != None):
            # Projection matrix
            fx = self.depth_intrin.fx
            fy = self.depth_intrin.fy
            cx = self.depth_intrin.ppx
            cy = self.depth_intrin.ppy

            apple_mask, branches_mask, metal_wire_mask, background_mask = detect_masks(
                model_path=MODEL_PATH, show_opt=False, image_obj=self.image_rgb
            )

            for point, color in zip(
                np.asarray(self.pcd.points), np.asarray(self.pcd.colors)
            ):
                x, y, z = point
                r, g, b = (color * 255).astype(np.uint8)
                u = int(fx * x / z + cx)
                v = int(fy * y / z + cy)
                if 0 <= u < width and 0 <= v < height:
                    image[v, u] = [b, g, r]
                if r != 0 and g != 0 and b != 0:
                    # this means that the point has been extracted correctly from the pointcloud
                    if apple_mask[v, u] != 0:
                        apple_points.append([x, y, z])
                    elif branches_mask[v, u] != 0:
                        branches_points.append([x, y, z])
                    elif metal_wire_mask[v, u] != 0:
                        wire_points.append([x, y, z])
                    elif background_mask[v, u] == 0:
                        # the point is not classified neither as apple, branch, metal wire or background
                        leaves_points.append([x, y, z])

            img = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.img_pub.publish(img)
            
            # img = self.bridge.compressed_imgmsg_to_cv2()

            # cv2.imshow("labeled image", image)
            # print('Press "q" to exit')
            # cv2.waitKey(self.timer_period*1000) & 0xFF == ord("q")

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.bitwise_and(apple_mask * 255, image_gray)

            
            if apple_points != []:
                self.apple_pub.publish(self.create_pointcloud(apple_points, "apple"))
            if branches_points != []:
                self.branches_pub.publish(
                    self.create_pointcloud(branches_points, "branch")
                )
            if wire_points != []:
                self.metal_wire_pub.publish(self.create_pointcloud(wire_points, "wire"))
            if leaves_points != []:
                self.leaves_pub.publish(self.create_pointcloud(leaves_points, "leaf"))

            self.get_logger().info("Published")

    def create_pointcloud(self, points, type):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        r = g = b = 0
        if type == "apple":
            r = 255
        if type == "branch":
            g = r = 255
            
        if type == "wire":
            b = 255
        if type == "leaf":
            g = 255

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_data = []

        for i in range(len(points)):
            point = points[i]
            # Encode the RGB value as UINT32
            rgb_uint32 = struct.unpack("I", struct.pack("BBBB", b, g, r, 0))[0]
            rgb_float32 = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb_float32])

        return create_cloud(header, fields, point_cloud_data)


def main(args=None):
    rclpy.init()
    detection = Detection()

    try:
        rclpy.spin(detection)
    except KeyboardInterrupt:
        detection.get_logger().info("Shutting down")

    detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
