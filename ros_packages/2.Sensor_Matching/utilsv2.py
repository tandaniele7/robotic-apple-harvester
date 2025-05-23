import numpy as np
import pyrealsense2 as rs
import struct
from geometry_msgs.msg import PointStamped, Quaternion, Vector3, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive 
from tf2_ros import TransformStamped
from tf2_geometry_msgs import do_transform_point

def create_collision_floor():
    box_object = CollisionObject()
    box_object.header.frame_id = "world"  # Or your desired frame
    box_object.id = "floor"

    # Define the box primitive
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.2, 1.2, 0.01]  # [height, radius]

    # Set the pose of the box
    box_pose = Pose()

    box_pose.position.x = 0.0
    box_pose.position.y = 0.0
    box_pose.position.z = -0.0051
    box_pose.orientation.w = 1.0

    # Add the primitive and pose to the collision object
    box_object.primitives = [box]
    box_object.primitive_poses = [box_pose]

    # Set the operation to add the object
    box_object.operation = CollisionObject.ADD 

    # Publish the collision object
    return box_object

def estimate_apple_radius(points, center_):
    # Calculate center of the apple in 3D space
    center = np.array(center_)
 
    # Calculate 3D distances from center to all points
    distances_3d = np.linalg.norm(points - center, axis=1)
 
    # Use the median of the top 10% distances as our radius estimate
    # This helps reduce the impact of outliers
    sorted_distances = np.sort(distances_3d)
    top_10_percent = sorted_distances[:int(0.2*len(sorted_distances))]
    radius = np.median(top_10_percent)
 
    # Convert to meters if necessary (assuming input is in millimeters)
    radius_meters = radius*0.8 #underestimation of radius to avoid problems with the gripper
 
    return radius_meters, center[2]

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
    return np.array([xr, yr, zr, tnr])
    
def float_to_rgb(float_rgb):
    packed = struct.pack("f", float_rgb)
    integers = struct.unpack("I", packed)[0]
    r = (integers >> 16) & 0xFF
    g = (integers >> 8) & 0xFF
    b = integers & 0xFF
    return r, g, b

def shift_point(quaternion, translation, shift_distance):
    # Create the transform from world to apple frame
    transform = TransformStamped()
    transform.transform.rotation = quaternion
    transform.transform.translation = translation


    

    # Shift the point in the apple frame (origin of apple frame)
    shifted_point_apple = PointStamped()
    shifted_point_apple.point.x = 0.0
    shifted_point_apple.point.y = 0.0
    shifted_point_apple.point.z = 0.0 + shift_distance

    # Create the inverse transform (apple to world)
    inverse_transform = TransformStamped()
    inverse_transform.transform.rotation = Quaternion(
        x=quaternion.x,
        y=quaternion.y,
        z=quaternion.z,
        w=quaternion.w
    )
    inverse_transform.transform.translation = Vector3(
        x=translation.x,
        y=translation.y,
        z=translation.z
    )

    # Transform the shifted point back to the world frame
    shifted_point_world = do_transform_point(shifted_point_apple, inverse_transform)

    return (shifted_point_world.point.x, shifted_point_world.point.y, float(translation.z))

