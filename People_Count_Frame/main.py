import cv2
import threading
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../Model/yolov8l.pt')  # Choose the model variant as needed

# RTSP stream URL
rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Desired frame dimensions (width, height)
frame_width = 1000
frame_height = 800

# Frame storage
frame = None

# Frame capture thread
def capture_frames():
    global frame
    while True:
        ret, captured_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from RTSP stream.")
            break
        frame = captured_frame


# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Frame counter
frame_counter = 0

roi_mask = None

roi_top_left = (200, 200)
roi_bottom_right = (600,500)

while True:
    if frame is not None:
        frame_counter += 1

        # Skip frames to reduce lag
        if frame_counter % 5 != 0:
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        # Create a transparent ROI overlay
        overlay = resized_frame.copy()
        alpha = 0.3  # Transparency factor

        # Draw the filled rectangle on the overlay
        cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0, resized_frame)

        # Create an ROI mask
        roi_mask = cv2.rectangle(
            np.zeros_like(resized_frame, dtype=np.uint8),
            roi_top_left,
            roi_bottom_right,
            (255, 255, 255),
            thickness=-1
        )

        # Apply the ROI mask to the frame using bitwise AND
        masked_frame = cv2.bitwise_and(resized_frame, roi_mask)

        # Run YOLOv8 inference on the masked frame
        results = model(masked_frame, conf=0.3, iou=0.4)

        # Initialize people count
        people_count = 0

        # Iterate through detected objects
        for box in results[0].boxes.data:
            class_id = int(box[5])
            if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
                people_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
                # Add label
                cv2.putText(resized_frame, "Alien", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)

        # Display the number of people detected on the frame
        cv2.putText(resized_frame, f"People count: {people_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the annotated frame
        cv2.imshow('People Count', resized_frame)

        # Break the loop on 'q' key press or window close
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
            break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
