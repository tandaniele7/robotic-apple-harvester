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
from sensor_msgs_py.point_cloud2 import create_cloud
from masks_detectionv2 import detect_masks, display_masks
from geometry_msgs.msg import Pose, Point, Transform, Quaternion, PointStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from apple_pose_srv.srv import GetApplePose
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformStamped, TransformBroadcaster
import argparse
from sklearn.decomposition import PCA
import tf2_geometry_msgs


parser = argparse.ArgumentParser(description="Vision Node [Choose between SIMULATION and REAL HARDWARE]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sim_mode", help = "Select between Simualtion mode [true] and Real Hardware [false]", action="store_true",)
args = parser.parse_args()
config = vars(args)


MODEL_PATH = "/workspace/src/Apple_segmentation_YOLOv8-daniele/new/1.Training/runs/segment/train3/weights/best.pt"

np.set_printoptions(threshold=sys.maxsize)

def estimate_apple_radius(points, intrinsics):
    # Calculate center of the apple in 3D space
    center = np.mean(points, axis=0)
 
    # Project 3D points to 2D image plane
    pixel_points = []
    for point in points:
        pixel = rs.rs2_project_point_to_pixel(intrinsics, point)
        pixel_points.append(pixel)
    pixel_points = np.array(pixel_points)
 
    # Project center to 2D image plane
    center_pixel = rs.rs2_project_point_to_pixel(intrinsics, center)
 
    # Calculate distances in pixel space
    distances = np.linalg.norm(pixel_points - center_pixel, axis=1)
 
    # Find the maximum distance in pixels
    max_distance_pixels = np.max(distances)
 
    # Convert pixel distance to 3D distance
    # We'll use the depth of the center point for this conversion
    depth = center[2]  # Assuming Z is depth
    max_distance_3d = max_distance_pixels * depth / intrinsics.fx
 
    # Radius is half of the maximum distance
    radius = max_distance_3d / 2
 
    return radius, depth


def calculate_grasping_orientation(tool_frame_position, apple_position):
    """
    Calculate the quaternion for grasping orientation.
    
    :param fixed_frame_position: np.array, the position of the fixed frame [x, y, z]
    :param apple_position: np.array, the position of the apple [x, y, z]
    :return: np.array, the quaternion [x, y, z, w]
    """
    # Calculate the direction vector from fixed frame to apple
    direction = tool_frame_position - apple_position
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Define the default up vector (assuming z is up in the fixed frame)
    up = np.array([0, 0, 1])
    
    # Calculate the rotation axis (cross product of up and direction)
    rotation_axis = np.cross(up, direction)
    
    # If rotation_axis is zero (direction is parallel to up), use x-axis as rotation axis
    if np.allclose(rotation_axis, 0):
        rotation_axis = np.array([1, 0, 0])
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot(up, direction))
    
    # Create a rotation object
    #r = R.from_rotvec(rotation_axis * rotation_angle)
    
    # Convert to quaternion
    #quaternion = r.as_quat()

    quaternion = np.array([np.sin(rotation_angle/2)*rotation_axis[0], np.sin(rotation_angle/2)*rotation_axis[1], np.sin(rotation_angle/2)*rotation_axis[2], np.cos(rotation_angle/2)])
    
    return quaternion


def multiply_quaternion(q1, q2):
    x1, y1, z1, tn1 = q1
    x2, y2, z2, tn2 = q2
    tnr = tn1*tn2 - x1*x2 - y1*y2 -z1*z2
    xr = tn1*x2 + tn2*x1 + y1*z2 -z1*y2
    yr = tn1*y2 + tn2*y1 - x1*z2 + z1*x2
    zr = tn1*z2 + tn2*z1 + x1*y2 - y1*x2
    return np.array([xr, yr, zr,tnr])
    
def float_to_rgb(float_rgb):
    packed = struct.pack("f", float_rgb)
    integers = struct.unpack("I", packed)[0]
    r = (integers >> 16) & 0xFF
    g = (integers >> 8) & 0xFF
    b = integers & 0xFF
    return r, g, b

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
        self.depth = []
        self.bridge = CvBridge()
        self.depth_intrin = None

        self.apple_ids = []
        self.apple_counter = 0  # counter for apple collision objects
        self.apple_count = 0    # counter for detected apples 

        # Used to get the coordinate of another frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = TransformBroadcaster(self)
        self.qtr = []

        # Publishers
        self.apples_pcl_pub = self.create_publisher(PointCloud2, apples_pcl_topic, 10)
        self.branches_pub = self.create_publisher(PointCloud2, branches_pcl_topic, 10)
        self.metal_wire_pub = self.create_publisher(PointCloud2, metalwire_pcl_topic, 10)
        self.collision_object_pub = self.create_publisher(CollisionObject, collision_object_topic, 10)
        self.apple_marker_pub = self.create_publisher(Marker, "apple_marker",10)

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

        self.get_logger().info("Vision Node starting...")

    def color_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.image_rgb = self.color

    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

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
        co.id = f"apple_{self.apple_counter}"
        self.apple_counter += 1

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

    def update_scene_callback(self, request, response):
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
        ):
            self.get_logger().info("Image and depth received")
            apple_masks, branches_mask, metal_wire_mask, background_mask = detect_masks(
                model_path=MODEL_PATH,
                show_opt=False,
                save_opt=False,
                image_obj=self.image_rgb,
                conf_apples=0.8,
                conf_branches=0.5,
                conf_metal_wire=0.7,
                conf_background=0.8,
            )

            
            apples_points, branches_points, wire_points, leaves_points = (
                self.depth_to_xyz(
                    self.depth,
                    self.depth_intrin,
                    apple_masks,
                    branches_mask,
                    metal_wire_mask,
                    background_mask,
                )
            )

            if list(apples_points) != []:
                for apple_index, apple_points in enumerate(apples_points):

                    #just for debugging
                    #self.apples_pcl_pub.publish(self.create_pointcloud(apple_points, "apple"))
                    if apple_points:  # Check if the apple_points list is not empty
                        apple_points_np = np.array(apple_points)
            
                        # Remove any points with infinity or NaN values
                        apple_points_np = apple_points_np[np.isfinite(apple_points_np).all(axis=1)]
            
                        if len(apple_points_np) > 0:
                            # Estimate radius and get distance to camera
                            radius, distance_to_camera = estimate_apple_radius(apple_points_np, self.depth_intrin)
            
                            # Calculate the center of the visible part of the apple
                            center = np.mean(apple_points_np, axis=0)
            
                            self.get_logger().info(f"Apple {apple_index} - Center: {center}, Estimated Radius: {radius}, Distance to Camera: {distance_to_camera}")
            
                            if radius >= 0.017:  # Adjust this threshold based on your specific setup
                                apple_info = {
                                    "apple_id": self.apple_count,
                                    "center": {
                                        "x": center[0],
                                        "y": center[1],
                                        "z": center[2],
                                    },
                                    "radius": radius,
                                    "distance_to_camera": distance_to_camera
                                }
            
                                self.apple_ids.append(apple_info)
                                self.apple_count += 1
                                
                                #apple_collision_object = self.create_apple_sphere(center, radius)
                                #self.collision_object_pub.publish(apple_collision_object)
            

                                
                                #self.get_logger().info(f"Apple {self.apple_count - 1} detected at {center} with radius {radius}")
                        
                if not self.apple_ids:
                    self.get_logger().warn("No apples detected that meet the minimum size criteria")
            

            if list(branches_points) != []:
                self.branches_pub.publish(self.create_pointcloud(branches_points, "branch"))
            if list(wire_points) != []:
                self.metal_wire_pub.publish(self.create_pointcloud(wire_points, "wire"))
            
            response.success = True
            response.message = ""
            
            self.get_logger().info("Scene Updated")
        else:
            response.success = False
            response.message = "Depth, Color or CameraInfo not correctly read by Server"
            self.get_logger().info(f"Scene NOT Updated: {response.message}")
            
        return response

    def get_apple_pose_callback(self, request, response):
        if self.apple_ids:
            response.pose.header.stamp = self.get_clock().now().to_msg()
            response.pose.header.frame_id = "world"
            apple_pts_world = self.transform_point([self.apple_ids[0]["center"]["x"], self.apple_ids[0]["center"]["y"], self.apple_ids[0]["center"]["z"]], "world",self.frame_id)
            
            x = apple_pts_world[0]
            y = apple_pts_world[1]
            z = apple_pts_world[2]
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
            marker.scale.x = marker.scale.y = marker.scale.z= radius*2    
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0    
        
            self.apple_marker_pub.publish(marker)

            tool_frame_point_world = self.transform_point([0.0, 0.0, 0.0], "world", "tool_frame")
            # tool_frame_quat = self.get_transform_quaternion("world", "tool_frame")
            self.qtr = calculate_grasping_orientation(tool_frame_position=tool_frame_point_world,apple_position=np.array([apple_pts_world[0], apple_pts_world[1], apple_pts_world[2]]))
            #self.qtr = [0.0, 0.0, 0.0, 1.0]
            
 

            rot_90_nz = [0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2]
            rot_90_nx = [-np.sqrt(2)/2,0.0, 0.0, np.sqrt(2)/2]

            rot_90_ny = [0.0,-np.sqrt(2)/2, 0.0, np.sqrt(2)/2]
            rot_90_x = [np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]
            rot_90_y = [0.0, np.sqrt(2)/2, 0.0, np.sqrt(2)/2]
            rot_180_z = [0.0, 0.0, 1.0, 0.0]
            rot_180_y = [0.0, 1.0, 0.0, 0.0]

            rot_180_x = [1.0, 0.0, 0.0, 0.0]


            self.qtr = multiply_quaternion(self.qtr, rot_180_y)

            response.pose.pose.orientation.x = self.qtr[0]
            response.pose.pose.orientation.y = self.qtr[1]
            response.pose.pose.orientation.z = self.qtr[2]
            response.pose.pose.orientation.w = self.qtr[3]
            
            self.broadcast_transform(apple_pose=apple_pts_world)            

            response.radius.data = self.apple_ids[0]["radius"]
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

    def depth_to_xyz(self, dpt, depth_intrin, apple_msks, branches_msk, wire_msk, background_msk):
        apples_pts = [[]]*len(apple_msks)
        branches_pts = []
        wire_pts = []
        leaves_pts = []
        for y in range(depth_intrin.height):
            for x in range(depth_intrin.width):
                if config["sim_mode"]:
                    dpth_ = dpt[y, x]
                else:
                    dpth_ = dpt[y, x] * 0.001  # Convert from mm to meters (ONLY WITH REALSENSE)
                if dpth_ > 0 and np.isfinite(dpth_):  # Check for positive and finite depth values
                    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dpth_)
                    if np.isfinite(point).all():  # Check if all coordinates are finite
                        for ii, msk in enumerate(apple_msks):
                            if msk[y, x] != 0:
                                apples_pts[ii].append(point)
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
            to_frame = 'world'
            
            # Get the transform from the from_frame to the to_frame
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time())
            
            # The origin of the 'from_frame' relative to 'to_frame' is simply the translation component of the transform
            translation = transform.transform.translation
            # self.get_logger().info(f"Origin of '{from_frame}' in '{to_frame}' frame: "
            #                       f"x={translation.x}, y={translation.y}, z={translation.z}")

            return np.array([translation.x, translation.y, translation.z])    
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")

    def broadcast_transform(self, apple_pose):
        # Create a TransformStamped message

        t = TransformStamped()
 
        # Set the timestamp to the current time
        t.header.stamp = self.get_clock().now().to_msg()
 
        # Set the frame_id (the parent frame)
        t.header.frame_id = 'world'  # The reference frame
 
        # Set the child_frame_id (the frame being published)
        t.child_frame_id = 'apple'  # The frame you are moving
 
        # Set the position of the child frame in relation to the parent frame
        t.transform.translation.x = apple_pose[0]  # x-coordinate in 'world' frame
        t.transform.translation.y = apple_pose[1]  # y-coordinate in 'world' frame
        t.transform.translation.z = apple_pose[2]  # z-coordinate in 'world' frame

 
        t.transform.rotation.x = self.qtr[0]
        t.transform.rotation.y = self.qtr[1]
        t.transform.rotation.z = self.qtr[2]
        t.transform.rotation.w = self.qtr[3]
 
        # Broadcast the transformation
        self.br.sendTransform(t)
    
    def transform_point(self, point, from_frame, to_frame):
        point_in_A = PointStamped()
        point_in_A.header.stamp = self.get_clock().now().to_msg()
        point_in_A.header.frame_id = from_frame  # The frame where the point is currently defined
        point_in_A.point.x = point[0]
        point_in_A.point.y = point[1]
        point_in_A.point.z = point[2]
 
        try:
            # Lookup the transformation from 'frame_A' to 'frame_B'
            transform = self.tf_buffer.lookup_transform(from_frame, to_frame, rclpy.time.Time())
 
            # Transform the point from 'frame_A' to 'frame_B'
            point_in_B = tf2_geometry_msgs.do_transform_point(point_in_A, transform)
            # Print the transformed point
            return [float(point_in_B.point.x), float(point_in_B.point.y), float(point_in_B.point.z)]
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


 

