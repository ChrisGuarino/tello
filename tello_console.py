import socket
import time

TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889
LOCAL_PORT = 9000

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", LOCAL_PORT))
sock.settimeout(5)

def send(cmd):
    """Send a command to the Tello drone and print the response."""
    try:
        sock.sendto(cmd.encode(), (TELLO_IP, TELLO_PORT))
        print(f"→ {cmd}")
        response, _ = sock.recvfrom(1024)
        print("←", response.decode())
    except socket.timeout:
        print("No response (timeout)")
    except Exception as e:
        print("Error:", e)

print("===================================")
print("     TELLO INTERACTIVE CONSOLE     ")
print("===================================")
print("Connecting…")
time.sleep(1)

# Enter SDK mode
send("command")

print("\nType Tello commands below.")
print("Examples: takeoff, land, battery?, forward 50, cw 90")
print("Type 'quit' or 'exit' to close.\n")

# Main loop
while True:
    cmd = input("> ").strip()

    if cmd.lower() in ("quit", "exit"):
        print("Exiting console.")
        break

    if cmd == "":
        continue

    send(cmd)

sock.close()
