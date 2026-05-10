#OAK-D Pro object detection.
# check these resources:
# https://docs.luxonis.com/software-v3/depthai/examples/detection_network/detection_network/
# https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/detection_network/
# https://docs.luxonis.com/software-v3/depthai/examples/
# https://docs.luxonis.com/software-v3/depthai


import depthai as dai
import cv2
import numpy as np
import time
import math

# FOV Constants (rad)
HFOV_RAD = math.radians(73.5)
VFOV_RAD = math.radians(41.0)

def frameNorm(frame, bbox):
    # Normalize bounding box coordinates to the frame dimensions.
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

pipeline = dai.Pipeline()

# Define camera sources 
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Stereo Depth Setup
stereo = pipeline.create(dai.node.StereoDepth)

# Linking the left and right mono streams to the stereo node
monoLeft.requestOutput((640, 400)).link(stereo.left)
monoRight.requestOutput((640, 400)).link(stereo.right)

# Spatial Detection Network - v3 Abstraction combining DetectionNetwork + SpatialLocationCalculator
modelDescription = dai.NNModelDescription("yolov6-nano") 
spatialDetectionNetwork = pipeline.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, modelDescription)

spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)  # Minimum Z distance in mm
spatialDetectionNetwork.setDepthUpperThreshold(5000) # Maximum Z distance in mm

# Receive classes (the label map) directly from the node
labelMap = spatialDetectionNetwork.getClasses()

# Host-side queues
qRgb = spatialDetectionNetwork.passthrough.createOutputQueue(maxSize=4, blocking=False)
qDet = spatialDetectionNetwork.out.createOutputQueue(maxSize=4, blocking=False)


pipeline.start()

print("Pipeline started! Press 'q' to quit.")

startTime = time.monotonic()
counter = 0

with pipeline:
    while pipeline.isRunning():
        # Pull synchronized frames and detections from the VPU
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None and inDet is not None:
            frame = inRgb.getCvFrame()
            detections = inDet.detections
            counter += 1

            for detection in detections:
                # Denormalize bounding box coordinates
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # Handle label mapping safely
                label = labelMap[detection.label] if labelMap and len(labelMap) > detection.label else str(detection.label)
                confidence = f"{int(detection.confidence * 100)}%"

                z_dist = detection.spatialCoordinates.z / 1000.0 # conversion from mm to meters

                # Calculate physical dimensions (width and height) for pedestrian clearance assessment
                norm_width = detection.xmax - detection.xmin
                norm_height = detection.ymax - detection.ymin
                
                physical_width = 2 * z_dist * math.tan(HFOV_RAD / 2) * norm_width
                physical_height = 2 * z_dist * math.tan(VFOV_RAD / 2) * norm_height

                # Showing bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
                # Showing dimension
                metrics_text = f"{label} {confidence} | Z: {z_dist:.2f}m | W: {physical_width:.2f}m H: {physical_height:.2f}m"
                cv2.putText(frame, metrics_text, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))

            # Showing FPS
            fps = counter / (time.monotonic() - startTime)
            cv2.putText(frame, f"NN FPS: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))

            cv2.imshow("OAK-D Pro Obstacle Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break
