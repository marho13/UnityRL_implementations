import socket, time

HOST = "10.0.0.112"#""127.0.0.1"
PORT = 11000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print("Starting...")
    s.connect((HOST, PORT))
    print("Connected")
    time.sleep(1)
    s.sendall(b"Boop<EOF>")
    print("Sent")
    s.close()
    time.sleep(2)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
    c.connect((HOST, PORT))
    print("Connected")
    time.sleep(1)
    c.sendall(b"this is the end<EOF>")
    print("Sent")
    c.close()