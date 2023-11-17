from pymavlink import mavutil
import time
import socket
the_connection = mavutil.mavlink_connection('udpin:localhost:14551')
the_connection.wait_heartbeat()
print("heart beat from system (system %u component %u)" %
      (the_connection.target_system, the_connection.target_component))
# Create a UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the port
server_address = ('127.0.0.1', 1234)  # Use your port number
server_socket.bind(server_address)
print("UDP server up and listening")
while True:
    data, address = server_socket.recvfrom(4096)  # buffer size is 4096 bytes
    print(f"received {len(data)} bytes from {address}")
    print(data.decode('utf-8'))
    vel = [float(i) for i in data.decode('utf-8').split(',')]
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        10, the_connection.target_system, the_connection.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, int(0b110111000111),
        0, 0, 0, vel[0], vel[1], vel[2], 0, 0, 0, 0, 0
    ))
    msg = the_connection.recv_match(type = 'LOCAL_POSITION_NED', blocking=True)
    print("vx: %.2f, vy: %.2f, vz: %.2f" % (msg.vx, msg.vy, msg.vz))