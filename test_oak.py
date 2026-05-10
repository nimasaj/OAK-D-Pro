# test OAK-D Pro while connected to make sure if it can stream with acceptable fps.

import depthai as dai
import cv2

pipeline = dai.Pipeline()

# Define sources using the unified Camera node
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Request 640x400 outputs
rgbOut = camRgb.requestOutput((640, 400), type=dai.ImgFrame.Type.RGB888p)
leftOut = monoLeft.requestOutput((640, 400))
rightOut = monoRight.requestOutput((640, 400))

# Linking Stereo & Depth streams
stereo = pipeline.create(dai.node.StereoDepth)
leftOut.link(stereo.left)
rightOut.link(stereo.right)

# Create the host-side queues before starting the pipeline
qRgb = rgbOut.createOutputQueue()
qDepth = stereo.disparity.createOutputQueue()

pipeline.start()

with pipeline:
    while pipeline.isRunning():
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is not None:
            cv2.imshow("RGB", inRgb.getCvFrame())

        if inDepth is not None:
            # Normalize the raw depth frame to 0-255 for visualization
            disp = inDepth.getFrame()
            disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype('uint8')
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
            cv2.imshow("Depth", disp)

        if cv2.waitKey(1) == ord('q'):
            break
