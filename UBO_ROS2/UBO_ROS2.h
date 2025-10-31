/**
 * \file UBO_ROS2.h
 * \author Jia Quan Loh
 * \brief The UBO_ROS2 class is a state machine for UBO robot with ROS2 communication
 * \date 2025-10-29
 *
 * \copyright Copyright (c) 2025
 *
 */
#ifndef UBO_SM_H
#define UBO_SM_H


#include "StateMachine.h"
#include "UBORobot.h"
#include "FLNLHelper.h"

// State Classes
#include "UBO_ROS2_States.h"
#include "UBO_ROS2_Node.h"

/**
 * @brief Example implementation of a StateMachine for the M3Robot class. States should implemented M3DemoState
 *
 */
class UBO_ROS2 : public StateMachine {

   public:
    UBO_ROS2(int argc, char **argv) ;
    ~UBO_ROS2();
    void init();
    void end();

    void hwStateUpdate();

    UBORobot *robot() { return static_cast<UBORobot*>(_robot.get()); } //!< Robot getter with specialised type (lifetime is managed by Base StateMachine)
    const std::shared_ptr<UBO_ROS2_Node> &get_node(){return m_Node;}
    std::shared_ptr<FLNLHelper> UIserver = nullptr;     //!< Pointer to communication server

    // //TODO: place in struct and pass to states (instead of whole state machine)
    double Command = 0;         //!< Command (state) currently applied
    // double MvtProgress = 0;     //!< Progress (status) along mvt
    // double Contribution = 0;    //!< User contribution to mvt
    // double MassComp =0;         //!< Mass comp value used for standard operations
    // Deweight_s DwData;

    private:
        std::shared_ptr<UBO_ROS2_Node> m_Node;
};

#endif /*M3_SM_H*/
