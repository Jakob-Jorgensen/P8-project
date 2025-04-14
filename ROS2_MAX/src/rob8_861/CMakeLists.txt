# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.5)

# Declare the package with a project name
project(rob8_861)  # This is the project name

# Find necessary dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate message interfaces for your custom message
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Command.msg"
  DEPENDENCIES std_msgs
  DEPENDENCIES builtin_interfaces
)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install Python packages
ament_python_install_package(${PROJECT_NAME})

# Install other resources or executable files (if needed)
install(PROGRAMS
  src/rob8_861/tcp_server.py 
  DESTINATION lib/${PROJECT_NAME}
)

# Register the package with ROS 2 and make it available to the ROS ecosystem
ament_package()

