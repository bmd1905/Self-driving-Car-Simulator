import socket


def create_socket():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the port on which you want to connect
    PORT = 54321

    # Connect to the server on local computer
    s.connect(('127.0.0.1', PORT))

    return s


def send_state(s, config):
    message_getState = bytes("0", "utf-8")
    s.sendall(message_getState)
    s.settimeout(0.1)  # Set a timeout of 0.1 second
    state_date = s.recv(100)

    current_speed, current_angle = state_date.decode(
        "utf-8"
    ).split(' ')
    message = bytes(f"1 {config.sendBack_angle} {config.sendBack_Speed}", "utf-8")
    s.sendall(message)

    return current_speed, current_angle


def recv_state(s):
    s.settimeout(0.2)  # Set a timeout of 0.2 second
    data = s.recv(100000)

    return data


