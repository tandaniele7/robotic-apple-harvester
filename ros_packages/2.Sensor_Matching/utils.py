import numpy as np
import pyrealsense2 as rs
import struct
from geometry_msgs.msg import PointStamped, Quaternion, Vector3
from tf2_ros import TransformStamped
from tf2_geometry_msgs import do_transform_point

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

    return (shifted_point_world.point.x, shifted_point_world.point.y, shifted_point_world.point.z)

