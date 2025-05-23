import rclpy
import cv2
import sys
import numpy as np
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from rclpy.qos import qos_profile_sensor_data
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
import pyrealsense2 as rs
from geometry_msgs.msg import Pose, Point, PointStamped, Quaternion, Vector3
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from apple_pose_srv.srv import GetApplePose
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformStamped, TransformBroadcaster
import argparse
from tf2_geometry_msgs import *
from sensor_msgs_py.point_cloud2 import create_cloud
from masks_detectionv3 import detect_masks, display_masks
from utilsv2 import (
    calculate_grasping_orientation,
    estimate_apple_radius,
    multiply_quaternion,
    shift_point,
    create_collision_floor
)


parser = argparse.ArgumentParser(
    description="Vision Node [Choose between SIMULATION and REAL HARDWARE]",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--sim_mode",
    help="Select between Simualtion mode [true] and Real Hardware [false]",
    action="store_true",
)
args = parser.parse_args()
config = vars(args)

ADAMW = "/workspace/src/Apple_segmentation_YOLOv8-daniele/new/1.Training/runs/segment/train100/weights/best.pt"
SGD = "/workspace/src/Apple_segmentation_YOLOv8-daniele/new/1.Training/runs/segment/train138/weights/best.pt"
MODEL_PATH = ADAMW
np.set_printoptions(threshold=sys.maxsize)


class Detection(Node):
    def __init__(self):
        super().__init__("vision_node")
        rclpy.logging.set_logger_level("VisionNode", 10)

        # Topics
        if not config["sim_mode"]:
            # real robot
            camera_info_depth_topic = "/camera/aligned_depth_to_color/camera_info"
            depth_topic = "/camera/aligned_depth_to_color/image_raw"
            color_topic = "/camera/color/image_raw"
        else:
            # simulation
            camera_info_depth_topic = "/wrist_mounted_camera/camera_info"
            depth_topic = "/wrist_mounted_camera/depth_image"
            color_topic = "/wrist_mounted_camera/image"

        apples_pcl_topic = "/pcl/apples"
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
        self.depth_frames_number = 1  # number of depth frames to be averaged
        self.depth_frames = [
            []
        ] * self.depth_frames_number  # depth frames that needs to be averaged to obtain a better acccuracy
        self.depth = []  # final depth (averaged)
        self.depth_ready = False  # this variable becomes true when the update_scene is requested and 5 frames are collected
        self.bridge = CvBridge()
        self.depth_intrin = None
        self.tool_frame_point_world = []

        self.update_scene_requested = False
        self.frames_counter = 0

        self.apple_ids = []
        self.apple_counter = 0  # counter for apple collision objects
        self.apple_count = 0  # counter for detected apples
        self.collision_pose = []

        # Used to get the coordinate of another frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = TransformBroadcaster(self)
        self.qtr = []

        # Publishers
        self.apples_pcl_pub = self.create_publisher(PointCloud2, apples_pcl_topic, 10)
        self.branches_pub = self.create_publisher(PointCloud2, branches_pcl_topic, 10)
        self.metal_wire_pub = self.create_publisher(
            PointCloud2, metalwire_pcl_topic, 10
        )
        self.collision_cylinder_publisher = self.create_publisher(
            CollisionObject, collision_object_topic, 10
        )
        self.apple_marker_pub = self.create_publisher(Marker, "apple_marker", 10)

        # Subscribers
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

        self.get_apple_pose_srv = self.create_service(
            GetApplePose, "get_apple_pose", self.get_apple_pose_callback
        )
        self.update_scene = self.create_service(
            Trigger, "update_scene", self.update_scene_callback
        )

        self.collision_cylinder_publisher.publish(create_collision_floor())
        
        self.get_logger().info("Vision Node starting...")

    def color_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.image_rgb = self.color

    def depth_callback(self, msg):
        self.depth_frames[self.frames_counter] = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )
        if self.update_scene_requested:
            if self.frames_counter == self.depth_frames_number - 1:
                # make average
                self.depth = np.mean(np.stack(self.depth_frames, axis=0), axis=0)
                self.depth_ready = True
                self.frames_counter = 0
            elif self.frames_counter < self.depth_frames_number - 1:
                self.frames_counter = self.frames_counter + 1
        else:
            self.frames_counter = 0
            self.depth_ready = False

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

    def collision_cylinder(self, mode_, center, *radius):
        cylinder_object = CollisionObject()
        cylinder_object.header.frame_id = "world"  # Or your desired frame
        cylinder_object.id = "cylinder"

        # Define the cylinder primitive
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [0.1, 0.03]  # [height, radius]

        # Set the pose of the cylinder
        cylinder_pose = Pose()
        if center != []:
            cylinder_pose.position.x = center[0]
            cylinder_pose.position.y = center[1]
            cylinder_pose.position.z = center[2] + 0.1
            cylinder_pose.orientation.w = 1.0

            # Add the primitive and pose to the collision object
            cylinder_object.primitives = [cylinder]
            cylinder_object.primitive_poses = [cylinder_pose]

            # Set the operation to add the object
            cylinder_object.operation = (
                CollisionObject.ADD if mode_ == "add" else CollisionObject.REMOVE
            )

            # Publish the collision object
            self.collision_cylinder_publisher.publish(cylinder_object)

    def update_scene_callback(self, request, response):
        self.update_scene_requested = True
        self.tool_frame_point_world = self.transform_point(
            [0.0, 0.0, 0.0], "world", "tool_frame"
        )

        apples_points = []
        branches_points = []
        wire_points = []
        leaves_points = []

        if (
            np.array(self.image_rgb).shape != ()
            and np.array(self.image_rgb).shape != (0, 0)
            and np.array(self.image_rgb).shape != (0,)
            and np.array(self.depth).shape != ()
            and np.array(self.depth).shape != (0, 0)
            and np.array(self.depth).shape != (0,)
            and self.depth_intrin is not None
            and self.depth_ready
        ):
            self.get_logger().info("Image and depth received")
            apple_masks, branches_mask, metal_wire_mask, background_mask = detect_masks(
                model_path=MODEL_PATH,
                show_opt=False,
                save_opt=True,
                image_obj=self.image_rgb,
                conf_apples=0.8,
                conf_branches=0.23,
                conf_metal_wire=0.7,
                conf_background=0.8,
                apple_mask_shrink_pixels=7,
            )

            apples_points, branches_points, wire_points, leaves_points = (
                self.depth_to_xyz(
                    self.depth,
                    self.depth_intrin,
                    apple_masks,
                    branches_mask,
                    metal_wire_mask,
                    background_mask,
                    min_depth=0.2,  # points closer than 30 are not inserted in the scene
                )
            )
            if apples_points != {}:
                for apple_index, apple_points in enumerate(apples_points):

                    # just for debugging
                    if apple_index == 0:
                        self.apples_pcl_pub.publish(
                            self.create_pointcloud(apples_points[apple_points], "apple")
                        )
                    if apples_points[apple_points]:  # Check if the apple_points list is not empty
                        apple_points_np = np.array(apples_points[apple_points])

                        # Remove any points with infinity or NaN values

                        apple_points_np = apple_points_np[
                            np.isfinite(apple_points_np).all(axis=1)
                        ]

                        if len(apple_points_np) > 0:

                            rot_90_z = [0.0, 0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]
                            rot_180_y = [0.0, 1.0, 0.0, 0.0]

                            # Estimate radius and get distance to camera
                            center = np.mean(apple_points_np, axis=0)
                            center_world = self.transform_point(
                                center, "world", self.frame_id
                            )

                            self.qtr = calculate_grasping_orientation(
                                tool_frame_position=self.tool_frame_point_world,
                                apple_position=np.array(
                                    [center_world[0], center_world[1], center_world[2]]
                                ),
                            )
                            self.qtr = multiply_quaternion(self.qtr, rot_180_y)
                            self.qtr = multiply_quaternion(self.qtr, rot_90_z)

                            shifted_appl_cntr = PointStamped()
                            
                            orientation = Quaternion(
                                x=self.qtr[0],
                                y=self.qtr[1],
                                z=self.qtr[2],
                                w=self.qtr[3],
                            )

                            shifted_appl_cntr = shift_point(
                                orientation,
                                Vector3(
                                    x=center_world[0],
                                    y=center_world[1],
                                    z=center_world[2],
                                ),
                                shift_distance=0.032
                            )

                            radius, distance_to_camera = estimate_apple_radius(
                                [self.transform_point(x, "world", self.frame_id) for x in apples_points[apple_points]], shifted_appl_cntr
                            )

                            # Calculate the center of the visible part of the apple

                            self.get_logger().info(
                                f"Apple {apple_index} - Center: {shifted_appl_cntr}, Estimated Radius: {radius}, Distance to Camera: {distance_to_camera}"
                            )

                            if (
                                radius >= 0.01
                            ):  # Adjust this threshold based on your specific setup

                                apple_info = {
                                    "apple_id": self.apple_count,
                                    "center": {
                                        "x": shifted_appl_cntr[0],
                                        "y": shifted_appl_cntr[1],
                                        "z": shifted_appl_cntr[2],
                                    },
                                    "orientation": orientation,
                                    "radius": radius,
                                    "distance_to_camera": distance_to_camera,
                                }

                                self.apple_ids.append(apple_info)
                                self.apple_count += 1

                                # self.get_logger().info(f"Apple {self.apple_count - 1} detected at {center} with radius {radius}")

                if not self.apple_ids:
                    self.get_logger().warn(
                        "No apples detected that meet the minimum size criteria"
                    )

            if list(branches_points) != []:
                self.branches_pub.publish(
                    self.create_pointcloud(branches_points, "branch")
                )
            if list(wire_points) != []:
                self.metal_wire_pub.publish(self.create_pointcloud(wire_points, "wire"))

            response.message = f"{len(self.apple_ids)}"
            response.success = True
            self.update_scene_requested = False

            self.get_logger().info("Scene Updated")
        elif not self.depth_ready:
            response.success = False
            response.message = "Depth frames not ready"
            self.get_logger().info(f"Scene NOT Updated: {response.message}")
        else:
            response.success = False
            response.message = "Depth, Color or CameraInfo not correctly read by Server"
            self.get_logger().info(f"Scene NOT Updated: {response.message}")

        return response

    def get_apple_pose_callback(self, request, response):
        self.collision_cylinder("remove", self.collision_pose)

        if self.apple_ids:
            response.pose.header.stamp = self.get_clock().now().to_msg()
            response.pose.header.frame_id = "world"

            x = self.apple_ids[0]["center"]["x"]
            y = self.apple_ids[0]["center"]["y"]
            z = self.apple_ids[0]["center"]["z"]
            radius = self.apple_ids[0]["radius"]

            response.pose.pose.position.x = x
            response.pose.pose.position.y = y
            response.pose.pose.position.z = z
            
            marker = Marker()
            marker.header.frame_id = "world"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = self.apple_counter
            marker.pose.position = Point(x=x, y=y, z=z)
            marker.scale.x = marker.scale.y = marker.scale.z = radius * 2
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.apple_marker_pub.publish(marker)

            response.pose.pose.orientation = self.apple_ids[0]["orientation"]

            # response.pose.pose.orientation.x = self.qtr[0]
            # response.pose.pose.orientation.y = self.qtr[1]
            # response.pose.pose.orientation.z = self.qtr[2]
            # response.pose.pose.orientation.w = self.qtr[3]

            # shifting the apple centre of 2 cm

            self.broadcast_transform(apple_pose=[x, y, z], qtrn=self.apple_ids[0]["orientation"])

            response.radius.data = self.apple_ids[0]["radius"]

            self.collision_pose = [x, y, z]
            self.collision_cylinder("add", self.collision_pose)

            response.message = ""
            response.success = True

            self.apple_ids.pop(0)
            self.get_logger().info(f"Apples in the queue {len(self.apple_ids)}")

        else:
            response.success = False
            response.message = "No apples in the queue"

        return response

    def create_pointcloud(self, points, type):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        if config["sim_mode"]:
            header.frame_id = "camera_color_frame"
        else:
            header.frame_id = self.frame_id
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

        for point in points:
            rgb_uint32 = struct.unpack("I", struct.pack("BBBB", b, g, r, 0))[0]
            rgb_float32 = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
            point_cloud_data.append([point[0], point[1], point[2], rgb_float32])

        return create_cloud(header, fields, point_cloud_data)

    def depth_to_xyz(
        self,
        dpt,
        depth_intrin,
        apple_msks,
        branches_msk,
        wire_msk,
        background_msk,
        min_depth,
    ):
        """
        This function creates the points that are associted to the different pointclouds in the
        case of branches, leaves and metal_wire, while it creates a list of list of points for every
        apple mask. Each apple mask has its own list of points.S
        S"""
        apples_pts = {}
        for i in range(len(apple_msks)):
            apples_pts[f"{i}"] = []

        branches_pts = []
        wire_pts = []
        leaves_pts = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                if config["sim_mode"]:
                    dpth_ = dpt[y, x]
                else:
                    dpth_ = (
                        dpt[y, x] * 0.001
                    )  # Convert from mm to meters (ONLY WITH REALSENSE)
                if (
                    dpth_ > 0 and np.isfinite(dpth_) and dpth_ > min_depth
                ):  # Check for positive and finite depth values
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dpth_)
                    if np.isfinite(point).all():  # Check if all coordinates are finite
                        for ii, msk in enumerate(apple_msks):
                            if msk[y, x] != 0:
                                apples_pts[f"{ii}"].append(point)
                        if branches_msk[y, x] != 0:
                            branches_pts.append(point)
                        elif wire_msk[y, x] != 0:
                            wire_pts.append(point)
                        elif background_msk[y, x] == 0:
                            leaves_pts.append(point)

        return (
            apples_pts,
            np.array(branches_pts),
            np.array(wire_pts),
            np.array(leaves_pts),
        )

    def get_tool_frame_origin(self):
        try:
            # Specify the frames between which you want the transform
            from_frame = "tool_frame"
            to_frame = "world"

            # Get the transform from the from_frame to the to_frame
            transform = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rclpy.time.Time()
            )

            # The origin of the 'from_frame' relative to 'to_frame' is simply the translation component of the transform
            translation = transform.transform.translation
            # self.get_logger().info(f"Origin of '{from_frame}' in '{to_frame}' frame: "
            #                       f"x={translation.x}, y={translation.y}, z={translation.z}")

            return np.array([translation.x, translation.y, translation.z])
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")

    def broadcast_transform(self, apple_pose, qtrn):
        # Create a TransformStamped message

        t = TransformStamped()

        # Set the timestamp to the current time
        t.header.stamp = self.get_clock().now().to_msg()

        # Set the frame_id (the parent frame)
        t.header.frame_id = "world"  # The reference frame

        # Set the child_frame_id (the frame being published)
        t.child_frame_id = "apple"  # The frame you are moving

        # Set the position of the child frame in relation to the parent frame
        t.transform.translation.x = apple_pose[0]  # x-coordinate in 'world' frame
        t.transform.translation.y = apple_pose[1]  # y-coordinate in 'world' frame
        t.transform.translation.z = apple_pose[2]  # z-coordinate in 'world' frame

        t.transform.rotation.x = qtrn.x
        t.transform.rotation.y = qtrn.y
        t.transform.rotation.z = qtrn.z
        t.transform.rotation.w = qtrn.w

        # Broadcast the transformation
        self.br.sendTransform(t)

    def transform_point(self, point, from_frame, to_frame):
        point_in_A = PointStamped()
        point_in_A.header.stamp = self.get_clock().now().to_msg()
        point_in_A.header.frame_id = (
            from_frame  # The frame where the point is currently defined
        )
        point_in_A.point.x = point[0]
        point_in_A.point.y = point[1]
        point_in_A.point.z = point[2]

        try:
            # Lookup the transformation from 'frame_A' to 'frame_B'
            transform = self.tf_buffer.lookup_transform(
                from_frame, to_frame, rclpy.time.Time()
            )

            # Transform the point from 'frame_A' to 'frame_B'
            point_in_B = tf2_geometry_msgs.do_transform_point(point_in_A, transform)
            # Print the transformed point
            return [
                float(point_in_B.point.x),
                float(point_in_B.point.y),
                float(point_in_B.point.z),
            ]
        except Exception as e:
            self.get_logger().error(f"Could not transform point: {str(e)}")


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