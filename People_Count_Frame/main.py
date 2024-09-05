# # main working code with roi
# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')  # Choose the model variant as needed
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
#
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     exit()
#
# # Desired frame dimensions (width, height)
# frame_width = 1000
# frame_height = 800
#
# # Frame storage
# frame = None
#
# # Frame capture thread
# def capture_frames():
#     global frame
#     while True:
#         ret, captured_frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#         frame = captured_frame
#
#
# # Start the frame capture thread
# capture_thread = threading.Thread(target=capture_frames, daemon=True)
# capture_thread.start()
#
# # Frame counter
# frame_counter = 0
#
# roi_mask = None
#
# roi_top_left = (200, 200)
# roi_bottom_right = (600,500)
#
# while True:
#     if frame is not None:
#         frame_counter += 1
#
#         # Skip frames to reduce lag
#         if frame_counter % 5 != 0:
#             continue
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # Create a transparent ROI overlay
#         overlay = resized_frame.copy()
#         alpha = 0.3  # Transparency factor
#
#         # Draw the filled rectangle on the overlay
#         cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle
#
#         # Blend the overlay with the original frame
#         cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0, resized_frame)
#
#         # Create an ROI mask
#         roi_mask = cv2.rectangle(
#             np.zeros_like(resized_frame, dtype=np.uint8),
#             roi_top_left,
#             roi_bottom_right,
#             (255, 255, 255),
#             thickness=-1
#         )
#
#         # Apply the ROI mask to the frame using bitwise AND
#         masked_frame = cv2.bitwise_and(resized_frame, roi_mask)
#
#         # Run YOLOv8 inference on the masked frame
#         results = model(masked_frame, conf=0.3, iou=0.4)
#
#         # Initialize people count
#         people_count = 0
#
#         # Iterate through detected objects
#         for box in results[0].boxes.data:
#             class_id = int(box[5])
#             if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
#                 people_count += 1
#                 # Draw bounding box
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
#                 # Add label
#                 cv2.putText(resized_frame, "Alien", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
#
#         # Display the number of people detected on the frame
#         cv2.putText(resized_frame, f"People count: {people_count}", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Show the annotated frame
#         cv2.imshow('People Count', resized_frame)
#
#         # Break the loop on 'q' key press or window close
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
#             break
#
# # Release the video capture object and close display window
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')  # Choose the model variant as needed
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
#
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     exit()
#
# # Desired frame dimensions (width, height)
# frame_width = 1000
# frame_height = 800
#
# # Frame storage
# frame = None
#
# # Global variables for ROI selection
# roi_defined = False
# roi_top_left = (0, 0)
# roi_bottom_right = (0, 0)
# drawing = False
#
# # Mouse callback function for drawing the ROI
# def draw_roi(event, x, y, flags, param):
#     global roi_top_left, roi_bottom_right, drawing, roi_defined
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         roi_top_left = (x, y)
#         roi_bottom_right = (x, y)
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             roi_bottom_right = (x, y)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         roi_bottom_right = (x, y)
#         roi_defined = True
#
# # Frame capture thread
# def capture_frames():
#     global frame
#     while True:
#         ret, captured_frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#         frame = captured_frame
#
# # Start the frame capture thread
# capture_thread = threading.Thread(target=capture_frames, daemon=True)
# capture_thread.start()
#
# # Set up the window and mouse callback for ROI selection
# cv2.namedWindow('People Count')
# cv2.setMouseCallback('People Count', draw_roi)
#
# # Frame counter
# frame_counter = 0
#
# while True:
#     if frame is not None:
#         frame_counter += 1
#
#         # Skip frames to reduce lag
#         if frame_counter % 5 != 0:
#             continue
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # If ROI is being defined, draw the rectangle
#         if drawing or roi_defined:
#             cv2.rectangle(resized_frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)
#
#         # Show the frame with the current ROI (if any)
#         cv2.imshow('People Count', resized_frame)
#
#         # Wait for the user to press 's' to start detection
#         if roi_defined and cv2.waitKey(1) & 0xFF == ord('s'):
#             print(f"ROI defined: {roi_top_left} to {roi_bottom_right}")
#             break
#
#         # Exit the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit()
#
# # Create an ROI mask based on the selected ROI
# roi_mask = np.zeros_like(resized_frame, dtype=np.uint8)
# cv2.rectangle(roi_mask, roi_top_left, roi_bottom_right, (255, 255, 255), thickness=-1)
#
# while True:
#     if frame is not None:
#         frame_counter += 1
#
#         # Skip frames to reduce lag
#         if frame_counter % 5 != 0:
#             continue
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # Apply the ROI mask to the frame using bitwise AND
#         imgRegion = cv2.bitwise_and(resized_frame, roi_mask)
#
#         # Run YOLOv8 inference on the masked frame
#         results = model(imgRegion, conf=0.3, iou=0.4)
#
#         # Initialize people count
#         people_count = 0
#
#         # Iterate through detected objects
#         for box in results[0].boxes.data:
#             class_id = int(box[5])
#             if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
#                 people_count += 1
#                 # Draw bounding box
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
#                 # Add label
#                 cv2.putText(resized_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
#
#         # Display the number of people detected on the frame
#         cv2.putText(resized_frame, f"People count: {people_count}", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Show the annotated frame
#         cv2.imshow('People Count', resized_frame)
#
#         # Break the loop on 'q' key press or window close
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
#             break
#
# # Release the video capture object and close display window
# cap.release()
# cv2.destroyAllWindows()

# user can draw the roi with free hand
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
frame_width = 1400
frame_height = 700

# Frame storage
frame = None

# Global variables for free-form ROI selection
points = []
drawing = False
roi_defined = False

# Mouse callback function for drawing the free-form ROI
def draw_freeform_roi(event, x, y, flags, param):
    global points, drawing, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]  # Start a new polygon

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        roi_defined = True

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

# Set up the window and mouse callback for free-form ROI selection
cv2.namedWindow('People Count')
cv2.setMouseCallback('People Count', draw_freeform_roi)

# Frame counter
frame_counter = 0

while True:
    if frame is not None:
        frame_counter += 1

        # Skip frames to reduce lag
        if frame_counter % 5 != 0:
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        # If ROI is being defined, draw the free-form polygon
        if drawing or roi_defined:
            if len(points) > 1:
                cv2.polylines(resized_frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Show the frame with the current ROI (if any)
        cv2.imshow('People Count', resized_frame)

        # Wait for the user to press 's' to start detection
        if roi_defined and cv2.waitKey(1) & 0xFF == ord('s'):
            print(f"ROI defined with {len(points)} points.")
            break

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Create an ROI mask based on the selected free-form ROI
roi_mask = np.zeros_like(resized_frame, dtype=np.uint8)
cv2.fillPoly(roi_mask, [np.array(points)], (255, 255, 255))

while True:
    if frame is not None:
        frame_counter += 1

        # Skip frames to reduce lag
        if frame_counter % 5 != 0:
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        # Apply the ROI mask to the frame using bitwise AND
        imgRegion = cv2.bitwise_and(resized_frame, roi_mask)

        # Run YOLOv8 inference on the masked frame
        results = model(imgRegion, conf=0.3, iou=0.4)

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
                cv2.putText(resized_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)

        # Display the number of people detected on the frame
        cv2.putText(resized_frame, f"People count: {people_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the selected ROI on the live stream
        if len(points) > 1:
            cv2.polylines(resized_frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Show the annotated frame
        cv2.imshow('People Count', resized_frame)

        # Break the loop on 'q' key press or window close
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
            break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()




