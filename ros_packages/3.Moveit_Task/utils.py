
import json
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
def write_json(dictionary):
    with open("test_result.json", "w") as outfile: 
        json.dump(dictionary, outfile)
        
def failed(starting_time, failure_time, test_dt, reason, atmp, appl_ps):
    x = appl_ps.pose.position.x
    y = appl_ps.pose.position.y
    z = appl_ps.pose.position.z
    
    data = {"attempt": atmp ,
        "starting_time": starting_time,
        "failure_time": failure_time - starting_time,
        "issue": reason,
        "success": False,
        "pose": [x, y, z]}
    test_dt.append(data)
    write_json(test_dt)
    return test_dt

def success(inference_time, motion_time, test_dt, atmp_, appl_ps_):
    x_ = appl_ps_.pose.position.x
    y_ = appl_ps_.pose.position.y
    z_ = appl_ps_.pose.position.z

    data = {"attempt": atmp_,
        "inference_time": inference_time,
        "motion_time": motion_time,
        # "cycle_time": success_time - starting_time,
        "success": True,
        "pose":[x_, y_, z_]}
    test_dt.append(data)
    write_json(test_dt)
    return test_dt

def create_collision_sphere(position, radius, mode):
    sphere_object = CollisionObject()
    sphere_object.header.frame_id = "world"  # Or your desired frame
    sphere_object.id = "apple"

    # Define the sphere primitive
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [radius]  # [radius]

    # Set the pose of the sphere
    sphere_pose = Pose()
    if position != []:
        sphere_pose.position.x = position[0]
        sphere_pose.position.y = position[1]
        sphere_pose.position.z = position[2]

        # Add the primitive and pose to the collision object
        sphere_object.primitives = [sphere]
        sphere_object.primitive_poses = [sphere_pose]

        # Set the operation to add the object
        sphere_object.operation = (
            CollisionObject.ADD if mode == "add" else CollisionObject.REMOVE
        )
    return sphere_object


