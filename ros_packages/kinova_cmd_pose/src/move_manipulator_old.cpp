#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <apple_pose_srv/srv/get_apple_pose.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <memory>
#include <thread>




class ApplePoseClient : public rclcpp::Node
{
public:
    ApplePoseClient() : Node("apple_pose_client")
    {
        client_apple_ = create_client<apple_pose_srv::srv::GetApplePose>("get_apple_pose");
        move_service_ = create_service<std_srvs::srv::Trigger>(
            "move_manipulator",
            std::bind(&ApplePoseClient::handle_move_request, this, std::placeholders::_1, std::placeholders::_2));

        
        RCLCPP_INFO(get_logger(), "Apple Pose Client node initialized. Waiting for 'move_manipulator' service calls.");
    }

    void handle_move_request(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request; // Unused parameter
        RCLCPP_INFO(get_logger(), "Received move_manipulator service call");

        if (!client_apple_->wait_for_service(std::chrono::seconds(4))) {
            RCLCPP_ERROR(get_logger(), "get_apple_pose service not available");
            response->success = false;
            response->message = "get_apple_pose service not available";
            return;
        }

        auto apple_pose_request = std::make_shared<apple_pose_srv::srv::GetApplePose::Request>();
        apple_pose_request->caller.data = "move";
        auto result_future = client_apple_->async_send_request(apple_pose_request,
            [this, response](rclcpp::Client<apple_pose_srv::srv::GetApplePose>::SharedFuture future) {
                auto result = future.get();
                if (result->success) {
                    RCLCPP_INFO(get_logger(), "Received apple pose. Attempting to move manipulator...");
                    bool move_success = move_manipulator(result->pose);
                    response->success = move_success;
                    response->message = move_success ? "Manipulator moved successfully" : "Failed to move manipulator";
                } else {
                    RCLCPP_ERROR(get_logger(), "Failed to get apple pose: %s", result->message.c_str());
                    response->success = false;
                    response->message = "Failed to get apple pose: " + result->message;
                }
                RCLCPP_INFO(get_logger(), "Service response: success=%s, message=%s", 
                            response->success ? "true" : "false", response->message.c_str());
            });

        RCLCPP_INFO(get_logger(), "Sent request to get_apple_pose service");
    }

    bool move_manipulator(const geometry_msgs::msg::PoseStamped& target_pose)
    {
        try {
            RCLCPP_INFO(get_logger(), "Initializing MoveGroupInterface");
            auto move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "manipulator");
            
            RCLCPP_INFO(get_logger(), "Setting pose target");
            move_group.setPoseTarget(target_pose, "tool_frame");
            //move_group.setOrientationTarget(target_orientation, "tool_frame");
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
    rclcpp::Client<apple_pose_srv::srv::GetApplePose>::SharedPtr client_apple_;
    
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr move_service_;
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
