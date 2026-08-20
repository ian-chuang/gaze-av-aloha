"""
Code to configure and control Interbotix grippers, including:
 - Register configuration
 - Current-limit setup
 - Trigger-based gripper command
"""
try:
    import rospy
except ImportError:
    rospy = None

try:
    from interbotix_xs_msgs.msg import JointSingleCommand
    from interbotix_xs_msgs.srv import (
        RegisterValues,
        RegisterValuesRequest,
    )
except ImportError:
    JointSingleCommand = None
    RegisterValues = None
    RegisterValuesRequest = None

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.5
GRIPPER_CURRENT_LIMIT = 100

# Reboot gripper motor.
def reboot_gripper(bot, sleep_time=1.0):
    if rospy is None:
        raise ImportError("rospy is required to reboot the gripper.")
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    rospy.sleep(sleep_time)

# Set a Dynamixel register for a single motor.
def set_register(robot_name, motor_name, reg_name, value):
    if rospy is None or RegisterValues is None or RegisterValuesRequest is None:
        raise ImportError("rospy and interbotix_xs_msgs are required to set gripper registers.")
    # print("Inside function set_register")
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)
    # print(f"Calling service {service_name} to set register {reg_name} to value {value}")

    req = RegisterValuesRequest()
    # print(f"Created RegisterValuesRequest: {req}")
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value
    # print(f"Request prepared: {req}")
    return srv(req)

# Configure gripper current limit and operating mode.
def configure_gripper(bot, robot_name):
    if rospy is None:
        raise ImportError("rospy is required to configure the gripper.")
    # print("Inside function configure_gripper")
    bot.dxl.robot_torque_enable("single", "gripper", False)
    # print("Torque disabled for gripper")
    rospy.sleep(0.2)
    
    set_register(robot_name, "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    # print(f"Setting gripper current limit to {GRIPPER_CURRENT_LIMIT}")
    rospy.sleep(0.2)

    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # print("Setting gripper operating mode to current-based position control")
    bot.dxl.robot_torque_enable("single", "gripper", True)
    # print("Torque enabled for gripper")
    rospy.sleep(0.5)

# Open or close the gripper based on trigger state.
def update_gripper(bot, trigger_pressed, close_position=GRIPPER_CLOSED, open_position=GRIPPER_OPEN):
    position = close_position if trigger_pressed else open_position
    command_gripper(bot, position)
    return position

# Directly send a gripper command.
def command_gripper(bot, position):
    if JointSingleCommand is None:
        raise ImportError("interbotix_xs_msgs is required to command the gripper.")
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = position
    bot.gripper.core.pub_single.publish(cmd)

# Open gripper
def open_gripper(bot):
    command_gripper(bot, GRIPPER_OPEN)

# Close gripper
def close_gripper(bot):
    command_gripper(bot, GRIPPER_CLOSED)

# Test gripper by sending open and close commands with a delay.
def test_gripper(bot, wait_time=1.0):
    if rospy is None:
        raise ImportError("rospy is required to test the gripper.")
    open_gripper(bot)
    rospy.sleep(wait_time)

    close_gripper(bot)
    rospy.sleep(wait_time)

    open_gripper(bot)
    rospy.sleep(wait_time)

# Read a register value from the gripper motor.
def get_register(robot_name, motor_name, reg_name):
    if rospy is None or RegisterValues is None or RegisterValuesRequest is None:
        raise ImportError("rospy and interbotix_xs_msgs are required to read gripper registers.")
    service_name = f"/{robot_name}/get_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name

    resp = srv(req)
    return list(resp.values)

# Print current gripper register values for debugging.
def print_gripper_registers(robot_name):
    try:
        current_limit = get_register(robot_name, "gripper", "Current_Limit")
        operating_mode = get_register(robot_name, "gripper", "Operating_Mode")
        torque_enable = get_register(robot_name, "gripper", "Torque_Enable")

        print(f"[{robot_name}] Current_Limit: {current_limit}")
        print(f"[{robot_name}] Operating_Mode: {operating_mode}")
        print(f"[{robot_name}] Torque_Enable: {torque_enable}")

    except Exception as e:
        print(f"[{robot_name}] Failed reading registers: {e}")
