#include "UBO_ROS2_Node.h"

UBO_ROS2_Node::UBO_ROS2_Node(const std::string &name, UBORobot *robot)
    : Node(name), m_Robot(robot)
{
    // Create an example subscription
    // m_Sub = create_subscription<sensor_msgs::msg::JointState>(
    //     "lfd_traj", 1000, std::bind(&UBO_ROS2_Node::lfd_traj_callback, this, _1)
    // );

    // Create a status publisher
    status_pub = create_publisher<std_msgs::msg::String>(
        "rbt_state", 1000
    );
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
UBO_ROS2_Node::get_interface()
{
    // Must have this method to return base interface for spinning
    return this->get_node_base_interface();
}

void
UBO_ROS2_Node::publish_state(std::string status)
{
    // Instantiate joint state message
    // spdlog::debug("publishing");
    std_msgs::msg::String msg;

    // Assign current header time stamp
    msg.data = status;

    // Publish the joint state message
    status_pub->publish(msg);
}



