import depthai as dai
import cv2
import time

pipeline = dai.Pipeline()

# Define camera (Color Camera)
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

# Request output at 640x400
rgbOut = camRgb.requestOutput((640, 400), type=dai.ImgFrame.Type.RGB888p)

# Create the host-side queue before starting the pipeline
qRgb = rgbOut.createOutputQueue()

# Start pipeline
pipeline.start()

start_time = time.time()
frame_count = 0
fps = 0

print("Starting camera stream. Press 'q' to quit.")

with pipeline:
    while pipeline.isRunning():
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

            # --- FPS CALCULATION ---
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0: # Update every 1 second
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time

            # Location to show FPS on the top-left corner of the stream
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("RGB Feed with FPS", frame)

        if cv2.waitKey(1) == ord('q'):
            break
