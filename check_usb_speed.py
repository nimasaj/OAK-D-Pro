import depthai as dai

print("Connecting to OAK-D camera...")

with dai.Device() as device:
    speed = device.getUsbSpeed()
    print(f"=================================")
    print(f"Connected USB Speed: {speed}")
    print(f"=================================")
    
    if speed == dai.UsbSpeed.HIGH:
        print("WARNING: You are connected at USB 2.0 speeds (480 Mbps).")
        print("Check your cable or ensure you are plugged into a blue USB 3.0 port.")
    elif speed in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
        print("SUCCESS: You are connected at USB 3.0 speeds (5 Gbps+).")
