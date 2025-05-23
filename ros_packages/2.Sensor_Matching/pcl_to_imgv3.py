import rclpy
import cv2
import sys
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
import pyrealsense2 as rs
from sensor_msgs_py.point_cloud2 import create_cloud, read_points
import open3d as o3d
from masks_detection import detect_masks
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from sklearn.cluster import DBSCAN

MODEL_PATH = "/workspace/src/Apple_segmentation_YOLOv8-daniele/1.Training/runs/segment/train3/weights/best.pt"

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
        super().__init__("sensor_matching_node")
        rclpy.logging.set_logger_level("PCL2imgTransformer", 10)

        # Topics
        camera_info_depth_topic = "/camera/aligned_depth_to_color/camera_info"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        pcl_topic = "/camera/depth/color/points"
        color_topic = "/camera/color/image_raw"
        apple_mask_topic = "/img/apple_mask"
        branches_pcl_topic = "/pcl/branches"
        metalwire_pcl_topic = "/pcl/metal_wire"
        leaves_pcl_topic = "/pcl/leaves"
        collision_object_topic = "/collision_object"

        self.frame_id = "camera_color_optical_frame"
        
        # Initialize variables
        self.pcl2 = None
        self.W = 640
        self.H = 480
        self.cam_info = None
        self.image_rgb = []
        self.depth = []
        self.bridge = CvBridge()
        self.depth_intrin = None
        self.pcd = o3d.geometry.PointCloud()

        self.apple_counter = 0
        
        # Publishers
        self.apple_mask_pub = self.create_publisher(Image, apple_mask_topic, 10)
        self.branches_pub = self.create_publisher(PointCloud2, branches_pcl_topic, 10)
        self.metal_wire_pub = self.create_publisher(PointCloud2, metalwire_pcl_topic, 10)
        self.leaves_pub = self.create_publisher(PointCloud2, leaves_pcl_topic, 10)
        self.collision_object_pub = self.create_publisher(CollisionObject, collision_object_topic, 10)

        # Subscribers
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

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile=qos_profile_sensor_data
        )
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            qos_profile=qos_profile_sensor_data,
        )

        # Timer
        self.timer_period = 10
        self.timer = self.create_timer(self.timer_period, self.spin_callback)
        self.get_logger().info("Initializing process")

    def color_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.image_rgb = self.color

    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def pcl_callback(self, msg):
        self.points = []
        for point in read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
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

    def create_apple_sphere(self, center, radius):
        co = CollisionObject()
        co.header.frame_id = self.frame_id
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = f"apple_{self.apple_counter}"  # Use the counter instead
        self.apple_counter += 1  # Increment the counter
        
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [radius]
        
        pose = Pose()
        pose.position.x = center[0]
        pose.position.y = center[1]
        pose.position.z = center[2]
        
        co.primitives.append(sphere)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        
        return co


    def spin_callback(self):
        width = self.W
        height = self.H
        apple_points = []
        branches_points = []
        wire_points = []
        leaves_points = []

        image = np.zeros((height, width, 3), dtype=np.uint8)

        if (np.array(self.image_rgb).shape != ()
            and np.array(self.image_rgb).shape != (0, 0)
            and np.array(self.image_rgb).shape != (0,)
            and np.array(self.depth).shape != ()
            and np.array(self.depth).shape != (0, 0)
            and np.array(self.depth).shape != (0,)
            and self.depth_intrin is not None):

            apple_mask, branches_mask, metal_wire_mask, background_mask = detect_masks(
                model_path=MODEL_PATH, show_opt=False, image_obj=self.image_rgb, conf_apples=0.85, conf_branches=0.6, 
                conf_metal_wire=0.68, conf_background=0.8
            )
            apple_points, branches_points, wire_points, leaves_points = self.depth_to_xyz(self.depth, self.depth_intrin, apple_mask, branches_mask, metal_wire_mask, background_mask)

            if list(apple_points)!=[]:
                apple_points_np = np.array(apple_points)
                clustering = DBSCAN(eps=0.05, min_samples=5).fit(apple_points_np)
                
                for cluster_id in set(clustering.labels_):
                    if cluster_id == -1:  # Skip noise points
                        continue
                    
                    cluster_points = apple_points_np[clustering.labels_ == cluster_id]
                    center = np.mean(cluster_points, axis=0)
                    radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
                    
                    apple_collision_object = self.create_apple_sphere(center, radius)
                    self.collision_object_pub.publish(apple_collision_object)

            if list(branches_points)!= []:
                self.branches_pub.publish(self.create_pointcloud(branches_points, "branch"))
            if list(wire_points)!= []:
                self.metal_wire_pub.publish(self.create_pointcloud(wire_points, "wire"))
            if list(leaves_points) != []:
                self.leaves_pub.publish(self.create_pointcloud(leaves_points, "leaf"))

            self.get_logger().info("Published")

    def create_pointcloud(self, points, type):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        r = g = b = 0
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

        for point in points:
            rgb_uint32 = struct.unpack("I", struct.pack("BBBB", b, g, r, 0))[0]
            rgb_float32 = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb_float32])

        return create_cloud(header, fields, point_cloud_data)
    
    def depth_to_xyz(self, dpt, depth_intrin, apple_msk, branches_msk, wire_msk, background_msk):
        apple_pts = []
        branches_pts = []
        wire_pts = []
        leaves_pts = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                dpth_ = dpt[y, x] * 0.001  # Convert from mm to meters
                if dpth_ > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dpth_)
                    if apple_msk[y, x] != 0:
                        apple_pts.append(point)
                    elif branches_msk[y, x] != 0:
                        branches_pts.append(point)
                    elif wire_msk[y, x] != 0:
                        wire_pts.append(point)
                    elif background_msk[y, x] == 0:
                        leaves_pts.append(point)

        return np.array(apple_pts), np.array(branches_pts), np.array(wire_pts), np.array(leaves_pts)

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

