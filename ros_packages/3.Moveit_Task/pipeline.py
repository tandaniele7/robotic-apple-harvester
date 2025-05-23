import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from apple_pose_srv.srv import GetApplePose  
from control_msgs.action import GripperCommand

class ApplePickingPipeline(Node):
    def __init__(self):
        super().__init__('apple_picking_pipeline')
        
        self.callback_group = ReentrantCallbackGroup()

        # Clients
        self.update_scene_client = self.create_client(Trigger, 'update_scene', callback_group=self.callback_group)
        self.get_apple_pose_client = self.create_client(GetApplePose, 'get_apple_pose', callback_group=self.callback_group)
        self.starting_position_client = self.create_client(Trigger, 'start_capturing', callback_group=self.callback_group)
        self.move_manipulator_client = self.create_client(Trigger, 'move_manipulator', callback_group=self.callback_group)
        self.place_apple_client = self.create_client(Trigger, 'place_apple', callback_group=self.callback_group)
        
        
        
        

        # Action Client
        self.gripper_command_client = ActionClient(self, GripperCommand, '/gen3_lite_2f_controller/gripper_cmd', callback_group=self.callback_group)
        

        self.planning_scene = None
        self.pipeline_state = 'SETUP'
        self.pipeline_timer = self.create_timer(4.0, self.run_pipeline_step, callback_group=self.callback_group)

    def update_scene(self):
        if not self.update_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/update_scene service not available, waiting again...')
            self.pipeline_state = 'HOME'
            return False
        
        req = Trigger.Request()
        future = self.update_scene_client.call_async(req)
        future.add_done_callback(self.update_scene_callback)

    def update_scene_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.pipeline_state = 'GET_APPLE_POSE'
            else:
                self.get_logger().warn(f'Failed to update scene: {result.message}')
                self.pipeline_state = 'HOME'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'HOME'

    def get_apple_pose(self):
        if not self.get_apple_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_apple_pose service not available, waiting again...')
            return
        
        req = GetApplePose.Request()
        future = self.get_apple_pose_client.call_async(req)
        future.add_done_callback(self.get_apple_pose_callback)

    def get_apple_pose_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.apple_pose = result.pose  # This should be a PoseStamped
                self.pipeline_state = 'PLAN_AND_EXECUTE'
            else:
                self.get_logger().warn(f'Failed to get apple pose: {result.message}')
                self.pipeline_state = 'INIT'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'INIT'
            
    def move_manipulator(self):
        if not self.move_manipulator_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/move_manipulator service not available, waiting again...')
            return
        
        req = Trigger.Request()
        future = self.move_manipulator_client.call_async(req)
        future.add_done_callback(self.move_manipulator_callback)

    def move_manipulator_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.pipeline_state = 'GRASP'   
            else:
                self.get_logger().warn(f'Failed to get move manipulator to Target Position: {result.message}')
                self.pipeline_state = 'GET_APPLE_POSE'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'GET_APPLE_POSE'
        
    def starting_position(self):
        if not self.starting_position_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Impossible to move to starting position: service not available, waiting again...')
            return False
        
        req = Trigger.Request()
        future = self.starting_position_client.call_async(req)
        future.add_done_callback(self.starting_position_callback)

    def starting_position_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.pipeline_state = 'INIT'
            else:
                #it keeps trying to go to starting position
                self.get_logger().warn('Failed to get to starting position')
                self.pipeline_state = 'HOME'
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')
            self.pipeline_state = 'HOME'

    def gripper_command(self, position):
        goal_msg = GripperCommand.Goal()
        if position == "close":
            goal_msg.command.position = 0.65
            goal_msg.command.max_effort = 0.0 
            
        elif position == "open":
            goal_msg.command.position = 0.05
            goal_msg.command.max_effort = 0.0 
            
        self.gripper_command_client.wait_for_server()

        self._send_goal_future = self.gripper_command_client.send_goal_async(goal_msg)

        self._send_goal_future.add_done_callback(self.gripper_command_response_callback)

    def gripper_command_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_command_result_callback)

    def gripper_command_result_callback(self, future):
        result = future.result().result
        if result.reached_goal == True:
            if self.pipeline_state == 'SETUP':
                self.pipeline_state = 'HOME'
            elif self.pipeline_state == 'GRASP':
                self.pipeline_state = 'PLACE'
        else:
            self.pipeline_state = 'SETUP'

    def place_apple(self):
        place_pose = PoseStamped()
        place_pose.header.frame_id = "base_link"
        place_pose.pose.position.x = 0.5  # Adjust as needed
        place_pose.pose.position.y = 0.0
        place_pose.pose.position.z = 0.5
        
        self.apple_pose = place_pose
        self.pipeline_state = 'PLAN_AND_EXECUTE'

    def run_pipeline_step(self):
        self.get_logger().info(f"Current pipeline state: {self.pipeline_state}")
        if self.pipeline_state == 'SETUP':
            self.gripper_command(position="open")
        elif self.pipeline_state == 'HOME':
            self.starting_position()
        elif self.pipeline_state == 'INIT':
            self.update_scene()
        elif self.pipeline_state == 'GET_APPLE_POSE':
            self.get_apple_pose()
        elif self.pipeline_state == 'PLAN_AND_EXECUTE':
            self.move_manipulator()
        elif self.pipeline_state == 'GRASP':
            self.gripper_command(position="close")
        elif self.pipeline_state == 'PLACE':
            self.place_apple()


def main(args=None):
    rclpy.init()
    pipeline = ApplePickingPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info("Shutting down")

    pipeline.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    """ rclpy.init(args=args)
    pipeline = ApplePickingPipeline()
    
    executor = MultiThreadedExecutor()
    executor.add_node(pipeline)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown() """