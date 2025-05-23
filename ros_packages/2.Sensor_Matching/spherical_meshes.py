from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial import ConvexHull

def create_spherical_meshes(points, frame_id="map"):
    collision_objects = []
    
    # Group points into clusters (you may need to implement or use a clustering algorithm here)
    clusters = cluster_points(points)
    
    for cluster in clusters:
        # Calculate the center of the cluster
        center = np.mean(cluster, axis=0)
        
        # Calculate the radius as the maximum distance from the center to any point
        radius = np.max(np.linalg.norm(cluster - center, axis=1))

        # the clustering of the points may fail, so it is better to add a check on the size of the apples
        if 0.05 <= 2*radius <= 0.12:
            # Create a CollisionObject
            co = CollisionObject()
            co.header.frame_id = frame_id
            co.id = f"sphere_{len(collision_objects)}"
            
            # Create a SolidPrimitive for the sphere
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [radius]
            
            # Set the pose of the sphere
            pose = Pose()
            pose.position = Point(x=center[0], y=center[1], z=center[2])
            pose.orientation.w = 1.0  # No rotation
            
            # Add the primitive to the collision object
            co.primitives = [sphere]
            co.primitive_poses = [pose]
            
            collision_objects.append(co)
    
    return collision_objects

def cluster_points(points, distance_threshold=0.1):
    # This is a simple clustering algorithm. You might want to use a more sophisticated one like DBSCAN.
    clusters = []
    for point in points:
        added_to_cluster = False
        for cluster in clusters:
            if np.min(np.linalg.norm(np.array(cluster) - point, axis=1)) < distance_threshold:
                cluster.append(point)
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([point])
    return clusters

# Usage example
def get_apple_meshes(apple_points, frame = 'map'):
    return create_spherical_meshes(apple_points, frame_id=frame)