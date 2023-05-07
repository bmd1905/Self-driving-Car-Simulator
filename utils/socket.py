import socket


def create_socket():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the port on which you want to connect
    PORT = 54321

    # Connect to the server on local computer
    s.connect(('127.0.0.1', PORT))

    return s

