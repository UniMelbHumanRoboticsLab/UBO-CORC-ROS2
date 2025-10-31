/**
 * \file UBO_ROS2_Node.h
 * \author Jia Quan Loh
 * \version 0.1
 * \date 2022-10-24
 * \copyright Copyright (c) 2022
 * \brief An example ROS2 node that also holds a reference to the robot object.
 */
#ifndef UBO_ROS2_Node_H
#define UBO_ROS2_Node_H

#include "UBORobot.h"
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

using std::placeholders::_1;

class UBO_ROS2_Node : public rclcpp::Node
{
public:
    UBO_ROS2_Node(const std::string &name, UBORobot *robot);

    void publish_state(std::string status);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_interface();


private:
    UBORobot *m_Robot;

    // rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_Sub;
    // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_Pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub;
    
};

#endif//UBO_ROS2_Node_H