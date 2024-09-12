# import cv2
# from ultralytics import YOLO
# from pathlib import Path
#
# # Load the YOLOv8 model (assuming you have best.pt trained for fire detection)
# model_path = 'best.pt'  # Replace with the path to your YOLOv8 best.pt model
# model = YOLO(model_path)
#
# # Set up the RTSP URL (replace with your actual RTSP stream)
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Define frame skipping interval
# frame_skip_interval = 3  # Process every Nth frame for smoother streaming
#
# # Define display size
# display_width = 1400  # Desired display width for resizing
# display_height = 700  # Desired display height for resizing
#
# # Function to perform fire detection from the RTSP stream
# def detect_fire_from_rtsp(rtsp_url):
#     # Open the RTSP stream
#     cap = cv2.VideoCapture(rtsp_url)
#
#     if not cap.isOpened():
#         print(f"Error: Unable to open RTSP stream at {rtsp_url}")
#         return
#
#     # Frame counter for skipping logic
#     frame_counter = 0
#
#     # Loop over the frames from the RTSP stream
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from RTSP stream")
#             break
#
#         # Skip frames based on the frame_skip_interval
#         frame_counter += 1
#         if frame_counter % frame_skip_interval != 0:
#             continue
#
#         # Perform inference on every Nth frame
#         results = model(frame)
#
#         # Annotate detected objects with class names and bounding boxes
#         annotated_frame = frame.copy()
#         if len(results) > 0 and len(results[0].boxes) > 0:
#             for result in results:
#                 for box in result.boxes:
#                     # Get class ID, confidence score, and bounding box coordinates
#                     class_id = int(box.cls[0])
#                     confidence = box.conf[0]
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#                     # Get the class name
#                     class_name = result.names[class_id]
#
#                     # Draw the bounding box and class name on the frame
#                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(annotated_frame, f'{class_name} ({confidence:.2f})',
#                                 (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#
#         # Resize the frame for display
#         resized_frame = cv2.resize(annotated_frame, (display_width, display_height))
#
#         # Display the resized frame
#         cv2.imshow('Object Detection (RTSP)', resized_frame)
#
#         # Press 'q' to quit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the stream and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Run object detection on the RTSP stream
# detect_fire_from_rtsp(rtsp_url)


# import cv2
# from ultralytics import YOLO
# from pathlib import Path
#
# # Load the YOLOv8 model (assuming you have best.pt trained for fire detection)
# model_path = 'best.pt'  # Replace with the path to your YOLOv8 best.pt model
# model = YOLO(model_path)
#
# # Set up the RTSP URL (replace with your actual RTSP stream)
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Define frame skipping interval
# frame_skip_interval = 3  # Process every Nth frame for smoother streaming
#
# # Define display size
# display_width = 1400  # Desired display width for resizing
# display_height = 700  # Desired display height for resizing
#
# # Function to perform fire detection from the RTSP stream
# def detect_fire_from_rtsp(rtsp_url):
#     # Open the RTSP stream
#     cap = cv2.VideoCapture(rtsp_url)
#
#     if not cap.isOpened():
#         print(f"Error: Unable to open RTSP stream at {rtsp_url}")
#         return
#
#     # Frame counter for skipping logic
#     frame_counter = 0
#
#     # Loop over the frames from the RTSP stream
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from RTSP stream")
#             break
#
#         # Skip frames based on the frame_skip_interval
#         frame_counter += 1
#         if frame_counter % frame_skip_interval != 0:
#             continue
#
#         # Perform inference on every Nth frame
#         results = model(frame)
#
#         # Annotate detected objects with class names and bounding boxes
#         annotated_frame = frame.copy()
#         if len(results) > 0 and len(results[0].boxes) > 0:
#             for result in results:
#                 for box in result.boxes:
#                     # Get class ID, confidence score, and bounding box coordinates
#                     class_id = int(box.cls[0])
#                     confidence = box.conf[0]
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#                     # Get the class name
#                     class_name = result.names[class_id]
#
#                     # Only display "fire" detections and ignore "smoke"
#                     if class_name.lower() == "fire":  # Adjust based on your class name for fire
#                         # Draw the bounding box and class name on the frame
#                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                         cv2.putText(annotated_frame, f'{class_name} ({confidence:.2f})',
#                                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#
#         # Resize the frame for display
#         resized_frame = cv2.resize(annotated_frame, (display_width, display_height))
#
#         # Display the resized frame
#         cv2.imshow('Fire Detection (RTSP)', resized_frame)
#
#         # Press 'q' to quit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the stream and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Run fire detection on the RTSP stream
# detect_fire_from_rtsp(rtsp_url)


import cv2
from ultralytics import YOLO
from pathlib import Path
from playsound import playsound
import threading

# Load the YOLOv8 model (assuming you have best.pt trained for fire detection)
model_path = 'best.pt'  # Replace with the path to your YOLOv8 best.pt model
model = YOLO(model_path)

# Set up the RTSP URL (replace with your actual RTSP stream)
rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'

# Define frame skipping interval
frame_skip_interval = 3  # Process every Nth frame for smoother streaming

# Define display size
display_width = 1400  # Desired display width for resizing
display_height = 700  # Desired display height for resizing

# Alarm sound path
alarm_sound_path = 'alarm.wav'  # Path to the alarm sound file

# Function to play the alarm sound in a separate thread
def play_alarm():
    try:
        playsound(alarm_sound_path)
    except Exception as e:
        print(f"Error playing alarm sound: {e}")

# Function to perform fire detection from the RTSP stream
def detect_fire_from_rtsp(rtsp_url):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Unable to open RTSP stream at {rtsp_url}")
        return

    # Frame counter for skipping logic
    frame_counter = 0

    # Alarm trigger flag to avoid continuous alarm sound
    alarm_triggered = False

    # Loop over the frames from the RTSP stream
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from RTSP stream")
            break

        # Skip frames based on the frame_skip_interval
        frame_counter += 1
        if frame_counter % frame_skip_interval != 0:
            continue

        # Perform inference on every Nth frame
        results = model(frame)

        # Annotate detected objects with class names and bounding boxes
        annotated_frame = frame.copy()
        fire_detected = False  # Flag to check if fire is detected in the current frame

        if len(results) > 0 and len(results[0].boxes) > 0:
            for result in results:
                for box in result.boxes:
                    # Get class ID, confidence score, and bounding box coordinates
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get the class name
                    class_name = result.names[class_id]

                    # Only display "fire" detections and ignore "smoke"
                    if class_name.lower() == "fire":  # Adjust based on your class name for fire
                        fire_detected = True
                        # Draw the bounding box and class name on the frame
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f'{class_name} ({confidence:.2f})',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Trigger the alarm if fire is detected and alarm is not already triggered
        if fire_detected and not alarm_triggered:
            alarm_triggered = True
            threading.Thread(target=play_alarm).start()  # Play the alarm sound in a separate thread

        # Resize the frame for display
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # Display the resized frame
        cv2.imshow('Fire Detection (RTSP)', resized_frame)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the stream and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run fire detection on the RTSP stream
detect_fire_from_rtsp(rtsp_url)
