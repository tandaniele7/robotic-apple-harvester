#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/motion_plan_response.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_srvs/srv/trigger.hpp>

class MotionPlanningNode : public rclcpp::Node
{
public:
  MotionPlanningNode() : Node("motion_planning_node")
  {
    // Initialize MoveIt2 interface
    move_group_interface_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      std::shared_ptr<rclcpp::Node>(this), "manipulator");

    // Create a service server for start_capturing
    start_capturing_server_ = this->create_service<std_srvs::srv::Trigger>(
      "start_capturing",
      std::bind(&MotionPlanningNode::startCapturingCallback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Motion Planning Node initialized");
  }

private:
  void startCapturingCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;  // Suppress unused parameter warning
    RCLCPP_INFO(this->get_logger(), "Received start_capturing request");

    // For this example, we'll use a hardcoded vector of joint values
    // In a real application, you would receive these values as input
    std::vector<double> joint_group_positions = {0.0, 1.9, 1.4, 1.57079632679, 1.8, -1.5693517502420085};

    // Ensure we have 6 joint values
    if (joint_group_positions.size() != 6)
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid number of joint positions. Expected 6, got %zu", joint_group_positions.size());
      response->success = false;
      response->message = "Invalid number of joint positions";
      return;
    }

    // Set the joint group positions
    move_group_interface_->setJointValueTarget(joint_group_positions);
    move_group_interface_->setMaxVelocityScalingFactor(0.8);

    // Create a plan to that target
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = (move_group_interface_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success)
    {
      RCLCPP_INFO(this->get_logger(), "Motion plan succeeded");

      // Execute the plan
      moveit::core::MoveItErrorCode execute_result = move_group_interface_->execute(my_plan);

      if (execute_result == moveit::core::MoveItErrorCode::SUCCESS)
      {
        RCLCPP_INFO(this->get_logger(), "Motion execution succeeded");
        response->success = true;
        response->message = "Motion plan and execution succeeded";
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Motion execution failed");
        response->success = false;
        response->message = "Motion plan succeeded but execution failed";
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Motion planning failed");
      response->success = false;
      response->message = "Motion planning failed";
    }
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_capturing_server_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MotionPlanningNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}