#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <apple_pose_srv/srv/move_manipulator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <memory>
#include <thread>




class ApplePoseClient : public rclcpp::Node
{
public:
    ApplePoseClient() : Node("move_manipulator")
    {
        move_service_ = create_service<apple_pose_srv::srv::MoveManipulator>(
            "move_manipulator",
            std::bind(&ApplePoseClient::handle_move_request, this, std::placeholders::_1, std::placeholders::_2));


        RCLCPP_INFO(get_logger(), "Apple Pose Client node initialized. Waiting for 'move_manipulator' service calls.");
    }

    void handle_move_request(
        const std::shared_ptr<apple_pose_srv::srv::MoveManipulator::Request> request,
        std::shared_ptr<apple_pose_srv::srv::MoveManipulator::Response> response)
    {
        //(void)request; // Unused parameter
        RCLCPP_INFO(get_logger(), "Received move_manipulator service call");

        geometry_msgs::msg::PoseStamped pose = request->pose;
        bool move_success = move_manipulator(pose);
        response->success = move_success;
        response->message = move_success ? "Manipulator moved successfully" : "Failed to move manipulator";
                
        RCLCPP_INFO(get_logger(), "Service response: success=%s, message=%s",
                    response->success ? "true" : "false", response->message.c_str());


    }

    bool move_manipulator(const geometry_msgs::msg::PoseStamped& target_pose)
    {
        try {
            RCLCPP_INFO(get_logger(), "Initializing MoveGroupInterface");
            auto move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "manipulator");

            RCLCPP_INFO(get_logger(), "Setting pose target");
            move_group.setPoseTarget(target_pose, "tool_frame");
            move_group.setMaxVelocityScalingFactor(0.8);

            RCLCPP_INFO(get_logger(), "Planning movement");
            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

            if (success) {
                RCLCPP_INFO(get_logger(), "Planning successful. Executing...");
                move_group.execute(my_plan);
                RCLCPP_INFO(get_logger(), "Execution complete");
                return true;
            } else {
                RCLCPP_ERROR(get_logger(), "Planning failed!");
                return false;
            }
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Exception in move_manipulator: %s", e.what());
            return false;
        }
    }


private:
    rclcpp::Service<apple_pose_srv::srv::MoveManipulator>::SharedPtr move_service_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ApplePoseClient>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}