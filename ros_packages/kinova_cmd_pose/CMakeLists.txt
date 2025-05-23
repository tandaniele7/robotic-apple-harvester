cmake_minimum_required(VERSION 3.8)
project(kinova_cmd_pose)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(apple_pose_srv REQUIRED)
find_package(moveit_msgs REQUIRED)

add_executable(move_manipulator src/move_manipulator.cpp)
add_executable(starting_position src/starting_position.cpp)
add_executable(apple_to_box src/apple_to_box.cpp)

ament_target_dependencies(move_manipulator rclcpp std_msgs moveit_ros_planning_interface geometry_msgs apple_pose_srv)
ament_target_dependencies(starting_position rclcpp moveit_ros_planning_interface moveit_msgs std_srvs)
ament_target_dependencies(apple_to_box rclcpp moveit_ros_planning_interface moveit_msgs std_srvs)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
install(TARGETS
  move_manipulator
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  starting_position
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  apple_to_box
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
