#include "UBO_ROS2.h"

using namespace std;

////////////////////////////////////////////////////////////////
// Transitions--------------------------------------------------
///////////////////////////////////////////////////////////////
bool isAPressed(StateMachine & sm) {
    UBO_ROS2 & SM = static_cast<UBO_ROS2 &>(sm);
    spdlog::trace("IsAPressed");
    if (SM.robot()->keyboard->getA() == true) {

        return true;
    }
    return false;
}
bool isSPressed(StateMachine & sm) {
    UBO_ROS2 & SM = static_cast<UBO_ROS2 &>(sm);
    if (SM.robot()->keyboard->getS() == true) {
        return true;
    }
    return false;
}

bool isCalibrationFinished(StateMachine & sm) {
    UBO_ROS2 & SM = static_cast<UBO_ROS2 &>(sm);
    return (SM.state<UBOCalibState>("calibState"))->isCalibDone();
}

//Exit CORC app properly
bool quit(StateMachine & SM) {
    UBO_ROS2 & sm = static_cast<UBO_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='Q' ) {
        std::raise(SIGTERM); //Clean exit
        return true;
    }

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("QUIT") ) {
        sm.UIserver->sendCmd(string("OKQU"));
        spdlog::debug("goToQuit");
        std::raise(SIGTERM); //Clean exit
        return true;
    }

    return false;
}

UBO_ROS2::UBO_ROS2(int argc, char **argv)  {
    //Create a Robot and set it to generic state machine
    setRobot(std::make_unique<UBORobot>());

    // Configure ROS2 initialisation options and disable SIGINT capture (handled by CORC)
    rclcpp::InitOptions ros_init = rclcpp::InitOptions();
    ros_init.shutdown_on_signal = false;
    rclcpp::init(argc, argv, ros_init);

    // Create the ROS2 node and pass a reference to the UBO object
    m_Node = std::make_shared<UBO_ROS2_Node>("UBO", robot());

    //Create state instances and add to the State Machine
    addState("initState", std::make_shared<UBOInitState>(robot(), this));
    addState("calibState", std::make_shared<UBOCalibState>(robot(), this));
    addState("idleState", std::make_shared<UBOIdleState>(robot(), this));
    addState("recordState", std::make_shared<UBORecordState>(robot(), this));

    //Define transitions between states
    // Transitions
    addTransition("initState", &isAPressed, "calibState");
    addTransition("calibState", &isCalibrationFinished, "idleState");
    addTransition("idleState", &isSPressed, "recordState");
    addTransition("recordState", &isSPressed, "idleState");

    //Initialize the state machine with first state of the designed state machine, using baseclass function.
    setInitState("initState");
    addTransitionFromAny(&quit, "idleState");
    addTransition("idleState", &quit, "idleState"); //From any does not apply to self (destination state)
}
UBO_ROS2::~UBO_ROS2() {
}

/**
 * \brief start function for running any designed statemachine specific functions
 * for example initialising robot objects.
 *
 */
void UBO_ROS2::init() {
    spdlog::debug("UBO_ROS2::init()");
    if(robot()->initialise()) {
        logHelper.initLogger("UBO_ROS2Log", "logs/UBO_ROS2.csv", LogFormat::CSV, true);
        logHelper.add(runningTime(), "Time (s)");
        logHelper.add(robot()->getUBO_readings(), "F");
        #ifdef NOROBOT
            UIserver = std::make_shared<FLNLHelper>(*robot(), "127.0.0.1");
            // UIserver = std::make_shared<FLNLHelper>(*robot(), "192.168.7.2");
        #else
            UIserver = std::make_shared<FLNLHelper>(*robot(), "127.0.0.1");
            // UIserver = std::make_shared<FLNLHelper>(*robot(), "192.168.7.2");
        #endif // NOROBOT
        UIserver->registerState(Command);
        // UIserver->registerState(MvtProgress);
        // UIserver->registerState(Contribution);
    }
    else {
        spdlog::critical("Failed robot initialisation. Exiting...");
        std::raise(SIGTERM); //Clean exit
    }
}
void UBO_ROS2::end() {
    if(running())
        UIserver->closeConnection();
    StateMachine::end();
}


/**
 * \brief Statemachine to hardware interface method. Run any hardware update methods
 * that need to run every program loop update cycle.
 *
 */
void UBO_ROS2::hwStateUpdate() {
    StateMachine::hwStateUpdate();
    //Also send robot state over network
    UIserver->sendState();
    //Attempt to reconnect (if not already waiting for connection)
    UIserver->reconnect();
    // Allow for the ROS2 node to execute callbacks (e.g., subscriptions)
    rclcpp::spin_some(get_node()->get_interface());
}


