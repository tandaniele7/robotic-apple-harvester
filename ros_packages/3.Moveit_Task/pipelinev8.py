import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, Empty
from apple_pose_srv.srv import GetApplePose, MoveManipulator
from control_msgs.action import GripperCommand
from threading import Lock
from rclpy.task import Future
import json

test_data = {}
    
 
class ApplePickingPipeline(Node):
    def __init__(self):
        super().__init__('apple_picking_pipeline')
 
        self.callback_group = ReentrantCallbackGroup()
 
        # Clients
        self.update_scene_client = self.create_client(Trigger, 'update_scene', callback_group=self.callback_group)
        self.get_apple_pose_client = self.create_client(GetApplePose, 'get_apple_pose', callback_group=self.callback_group)
        self.starting_position_client = self.create_client(Trigger, 'start_capturing', callback_group=self.callback_group)
        self.move_manipulator_client = self.create_client(MoveManipulator, 'move_manipulator', callback_group=self.callback_group)
        self.place_apple_client = self.create_client(Trigger, 'apple_to_box', callback_group=self.callback_group)
        self.clear_octomap_client = self.create_client(Empty, 'clear_octomap', callback_group=self.callback_group)
        
        
        self.starting_time = 0
        self.cycle_time = 0
        self.apples_queue = 0
 
        # Action Client
        self.gripper_command_client = ActionClient(self, GripperCommand, '/gen3_lite_2f_controller/gripper_cmd', callback_group=self.callback_group)
 
        self.planning_scene = None
        self.pipeline_state = 'SETUP'
        self.pipeline_timer = self.create_timer(0.5, self.run_pipeline_step, callback_group=self.callback_group)
 
        self.lock = Lock()
        self.current_step_future = None
 
    def run_pipeline_step(self):
        with self.lock:
            if self.current_step_future and not self.current_step_future.done():
                #self.get_logger().info("Previous step still in progress. Skipping.")
                return
 
            self.get_logger().info(f"Current pipeline state: {self.pipeline_state}")
 
            if self.pipeline_state == 'SETUP':
                self.starting_time = self.get_clock().now().nanoseconds
                self.current_step_future = self.gripper_command(position="open")
            elif self.pipeline_state == 'HOME':
                self.current_step_future = self.starting_position()
            elif self.pipeline_state == 'INIT':
                self.starting_time = self.get_clock().now().nanoseconds
                self.current_step_future = self.update_scene()
            elif self.pipeline_state == 'GET_APPLE_POSE':
                self.current_step_future = self.get_apple_pose()
            elif self.pipeline_state == 'PLAN_AND_EXECUTE':
                self.current_step_future = self.move_manipulator()
            elif self.pipeline_state == 'GRASP':
                self.current_step_future = self.gripper_command(position="close")
            elif self.pipeline_state == 'PLACE':
                self.cycle_time = (self.get_clock().now().nanoseconds - self.starting_time) * 1e-9
                print(self.cycle_time)
                self.current_step_future = self.place_apple()
 
    def update_scene(self):
        future = Future()
        req = Empty.Request()
        self.clear_octomap_client.call_async(req)
        if not self.update_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/update_scene service not available, waiting again...')
            self.pipeline_state = 'HOME'
            future.set_result(None)
            return future
 
        req = Trigger.Request()
        call_future = self.update_scene_client.call_async(req)
        call_future.add_done_callback(lambda f: self.update_scene_callback(f, future))
        return future
 
    def update_scene_callback(self, call_future, future):
        try:
            result = call_future.result()
            if result.success:
                self.apples_queue = int(result.message)
                self.pipeline_state = 'GET_APPLE_POSE'
            else:
                self.get_logger().warn(f'Failed to update scene: {result.message}')
                self.pipeline_state = 'HOME'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'HOME'
        finally:
            future.set_result(None)
 
    def get_apple_pose(self):
        future = Future()
        if not self.get_apple_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_apple_pose service not available, waiting again...')
            future.set_result(None)
            return future

        req = GetApplePose.Request()
        call_future = self.get_apple_pose_client.call_async(req)
        call_future.add_done_callback(lambda f: self.get_apple_pose_callback(f, future))
        return future
 
    def get_apple_pose_callback(self, call_future, future):
        try:
            result = call_future.result()
            if result.success:
                self.apple_pose = result.pose  # This should be a PoseStamped
                self.apples_queue = self.apples_queue - 1
                self.pipeline_state = 'PLAN_AND_EXECUTE'
            else:
                self.get_logger().warn(f'Failed to get apple pose: {result.message}')
                self.pipeline_state = 'INIT'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'INIT'
        finally:
            future.set_result(None)
 
    def move_manipulator(self):
        future = Future()
        if not self.move_manipulator_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('/move_manipulator service not available, waiting again...')
            future.set_result(None)
            return future
 
        req = MoveManipulator.Request()
        req.pose = self.apple_pose
        call_future = self.move_manipulator_client.call_async(req)
        call_future.add_done_callback(lambda f: self.move_manipulator_callback(f, future))
        return future
 
    def move_manipulator_callback(self, call_future, future):
        try:
            result = call_future.result()
            if result.success:
                self.pipeline_state = 'GRASP'   
            else:
                self.get_logger().warn(f'Failed to move manipulator to Target Position: {result.message}')
                self.pipeline_state = 'GET_APPLE_POSE'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'GET_APPLE_POSE'
        finally:
            future.set_result(None)
 
    def starting_position(self):
        future = Future()
        if not self.starting_position_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Impossible to move to starting position: service not available, waiting again...')
            future.set_result(None)
            return future
 
        req = Trigger.Request()
        call_future = self.starting_position_client.call_async(req)
        call_future.add_done_callback(lambda f: self.starting_position_callback(f, future))
        return future
 
    def starting_position_callback(self, call_future, future):
        try:
            result = call_future.result()
            if result.success:
                self.pipeline_state = 'INIT'
            else:
                #it keeps trying to go to starting position
                self.get_logger().warn('Failed to get to starting position')
                self.pipeline_state = 'HOME'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'HOME'
        finally:
            future.set_result(None)
 
    def gripper_command(self, position):
        future = Future()
        goal_msg = GripperCommand.Goal()
        if position == "close":
            goal_msg.command.position = 0.43
            goal_msg.command.max_effort = 18.0 
        elif position == "open":
            goal_msg.command.position = 0.05
            goal_msg.command.max_effort = 0.0 
 
        self.gripper_command_client.wait_for_server()
 
        send_goal_future = self.gripper_command_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(lambda f: self.gripper_command_response_callback(f, future))
        return future
 
    def gripper_command_response_callback(self, send_goal_future, future):
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            future.set_result(False)
            return
 
        self.get_logger().info('Goal accepted :)')
 
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(lambda f: self.gripper_command_result_callback(f, future))
 
    def gripper_command_result_callback(self, get_result_future, future):
        result = get_result_future.result().result
        if result.reached_goal == True:
            if self.pipeline_state == 'SETUP':
                if self.apples_queue > 0:
                    self.pipeline_state = 'GET_APPLE_POSE'
                else:
                    self.pipeline_state = 'HOME'
            elif self.pipeline_state == 'GRASP':
                self.pipeline_state = 'PLACE'
            future.set_result(True)
        else:
            self.pipeline_state = 'SETUP'
            future.set_result(False)
 
    def place_apple(self):
        future = Future()
        if not self.place_apple_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Impossible to move to starting position: service not available, waiting again...')
            future.set_result(None)
            return future
 
        req = Trigger.Request()
        call_future = self.place_apple_client.call_async(req)
        call_future.add_done_callback(lambda f: self.place_apple_callback(f, future))
        return future
 
    def place_apple_callback(self, call_future, future):
        try:
            result = call_future.result()
            if result.success:
                self.pipeline_state = 'SETUP'
            else:
                #it keeps trying to go to starting position
                self.get_logger().warn('Failed to place apple to final box')
                self.pipeline_state = 'SETUP'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'SETUP'
        finally:
            future.set_result(None)

def main(args=None):
    rclpy.init(args=args)
    pipeline = ApplePickingPipeline()
 
    executor = MultiThreadedExecutor()
    executor.add_node(pipeline)
 
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()