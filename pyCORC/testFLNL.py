import time
from FLNL import *
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

client = FLNLClient()

ip="127.0.0.1" #"192.168.8.1" BBAI default Wifi IP address
client.Connect(ip=ip, port=2048)

while not client.IsConnected():
	print("Waiting for connection...")
	time.sleep(1)

while not client.IsCmd("rec"): 
	print("waiting for rec")
	time.sleep(0.5)
while True:
	if client.IsValues():
		wrench = client.GetValues()
		print(f"time:{wrench[0]}")
		for i in range(3):
			print(f"RFT {i}: {wrench[i*6+1:i*6+6+1]}")
		# print("recording...")

client.Close()