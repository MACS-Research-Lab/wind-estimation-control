from pymavlink import mavutil
import time

the_connection = mavutil.mavlink_connection('udpin:localhost:14551')

the_connection.wait_heartbeat()
print("heart beat from system (system %u component %u)" % 
      (the_connection.target_system, the_connection.target_component))

the_connection.mav.command_long_send(
    the_connection.target_system, the_connection.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 4, 0, 0, 0, 0, 0, 0)

the_connection.mav.command_long_send(
    the_connection.target_system, the_connection.target_component, 
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)

time.sleep(2)

the_connection.mav.command_long_send(
    the_connection.target_system, the_connection.target_component, 
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 30)

msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)