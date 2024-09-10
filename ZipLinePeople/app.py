# working in roi with detection of multiple cameras
from flask import Flask, request, jsonify
import numpy as np
import cv2
import math
from ultralytics import YOLO
from sort import Sort
from multiprocessing import Process, Event
import paho.mqtt.client as mqtt
import json
import os
import time

app = Flask(__name__)

# Global variables
mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
mqtt_port = 1883
mqtt_topic = "zipline/detected"

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)
mqtt_client.loop_start()  # Non-blocking MQTT loop

# MQTT publish function
def publish_message(motion_type, rtsp_link, site_id, camera_id, alarm_id, people_count,image):
    message = {
        "rtsp_link": rtsp_link,
        "siteId": site_id,
        "cameraId": camera_id,
        "alarmId":alarm_id,
        "type": motion_type,
        "people_count": people_count,
        "image": image
    }
    mqtt_client.publish(mqtt_topic, json.dumps(message))
    print(f"Published message: {json.dumps(message)}")


# Initialize tracker and model
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
model = YOLO("../Model/yolov8l.pt")

# Dictionary to store ongoing processes and stop events
detection_processes = {}

# Function to apply the given offsets to points in the JSON data
def use_coordinates(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    placed_points = []
    for point in points:
        placed_x = int(point[0] + x_offset)
        placed_y = int(point[1] + y_offset)
        placed_points.append((placed_x, placed_y))

    return placed_points


# Check if the centroid is inside the ROI polygon
def is_inside_roi(cx, cy, roi_points):
    result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
    return result >= 0

# Ensure directories exist
image_dir = "images"
# video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
# os.makedirs(video_dir, exist_ok=True)


def capture_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(image_dir, f"motion_{timestamp}.jpg")
    cv2.imwrite(image_filename, frame)
    absolute_image_path = os.path.abspath(image_filename)
    print(f"Captured image: {absolute_image_path}")
    return absolute_image_path

# People counting logic based on ROI and zipline
def detect_people_count(rtsp_url, site_id, camera_id, alarm_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    count = 0  # Single count variable for both up and down
    frame_skip = 3  # To process every frame (you can adjust this based on performance needs)
    frame_count = 0

    # Dictionary to store the previous Y position (cy) of each person to detect crossings
    previous_positions = {}

    # Debounce mechanism to avoid rapid re-counting
    debounce_time = 1.5  # seconds
    last_count_time = {}

    display_width, display_height = display_size

    # Define a buffer zone around the zipline
    buffer_vertical = 20  # Vertical buffer around the line
    buffer_horizontal =20  # Horizontal buffer around the line

    # Arrow start and end points based on arrow coordinates
    arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
    arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))

    # Determine the direction of the arrow (up or down)
    arrow_direction = arrow_coords['points'][1][1]  # Positive for down, negative for up

    # Create a window for each camera stream
    window_name = f"Camera {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set the window size to the display size provided
    cv2.resizeWindow(window_name, display_width, display_height)

    # Error handling for the camera stream
    if not cap.isOpened():
        print(f"Error: Unable to open camera stream for camera {camera_id}")
        return

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            print(f"Error: Failed to read from camera {camera_id}")
            break

        # Resize the frame to match the provided display size
        frame = cv2.resize(frame, (display_width, display_height))

        # Draw zipline and ROI on the frame using the coordinates directly from the JSON
        cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)  # Draw the zipline
        cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)  # Draw the ROI

        # Draw the arrow on the frame
        cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)  # Draw the arrow

        # Display the frame with the visual indicators (arrow, zipline, ROI)
        cv2.imshow(window_name, frame)

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Create an ROI mask
            roi_mask = np.zeros_like(frame[:, :, 0])  # Same size as one channel of the frame
            cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)

            # Apply the ROI mask to the frame
            roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

            # Run YOLO model on the cropped frame (inside the ROI)
            results = model(roi_frame, stream=True)
            detections = np.empty((0, 5))

            # Start tracking only when people are detected inside the ROI
            tracking_started = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1b, y1b, x2b, y2b = box.xyxy[0]
                    x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)

                    # Confidence and class filtering (person class)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
                        detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
                        tracking_started = True  # Begin tracking when a person is detected inside the ROI

            # Proceed to tracking only if a person is detected inside the ROI
            if tracking_started:
                tracked_people = tracker.update(detections)

                # Get the slope (m) and intercept (b) of the zipline
                m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
                b = zipline_coords[0][1] - m * zipline_coords[0][0]

                current_time = cv2.getTickCount() / cv2.getTickFrequency()

                for person in tracked_people:
                    x1, y1, x2, y2, person_id = person
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calculate the centroid of the bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box

                    # Check if the centroid is inside the ROI
                    if not is_inside_roi(cx, cy, roi_coords):
                        continue  # Skip if the person is outside the ROI

                    # Calculate the y-coordinate of the zipline at the person's x-position (cx)
                    zipline_y = m * cx + b

                    # Get the previous y-coordinate of the person (if any)
                    prev_cy = previous_positions.get(person_id, cy)

                    # Restrict detection to objects within the buffer zone of the zipline
                    if (abs(cy - zipline_y) <= buffer_vertical) and (zipline_coords[0][0] - buffer_horizontal <= cx <= zipline_coords[1][0] + buffer_horizontal):
                        # Check debounce time to avoid re-counting too quickly
                        if person_id not in last_count_time or (current_time - last_count_time[person_id]) > debounce_time:
                            # If the arrow points up (negative value), only count upward crossings
                            if arrow_direction < 0 and cy < zipline_y and prev_cy >= zipline_y:
                                count += 1
                                last_count_time[person_id] = current_time
                                frame_copy = frame.copy()
                                image_filename = capture_image(frame_copy)
                                # capture_image(frame, camera_id, count)  # Save the frame
                                publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, count,image_filename)
                                cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 255, 0), 5)  # Change line color to green for up

                            # If the arrow points down (positive value), only count downward crossings
                            elif arrow_direction > 0 and cy > zipline_y and prev_cy <= zipline_y:
                                count += 1
                                last_count_time[person_id] = current_time
                                frame_copy = frame.copy()
                                image_filename = capture_image(frame_copy)
                                # capture_image(frame, camera_id, count)  # Save the frame
                                publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, count,image_filename)
                                cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)  # Keep line color red for down

                    # Update the previous position of the person
                    previous_positions[person_id] = cy

                    # Draw bounding box and centroid on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                # Show people count on the frame
                cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the updated frame in the window for the camera
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()

    return count




# Function to start each camera in its own process
def start_camera_process(rtsp_url,site_id, camera_id,alarm_id,roi_coords, zipline_coords, arrow_coords, display_size):
    stop_event = Event()
    detection_process = Process(target=detect_people_count, args=(
        rtsp_url,site_id, camera_id,alarm_id, roi_coords, zipline_coords, arrow_coords, display_size,stop_event))
    detection_process.start()
    return detection_process, stop_event


# Flask route to start people counting detection for multiple cameras
@app.route('/zipline_start', methods=['POST'])
def start_counting():
    data = request.json

    if not isinstance(data, list):
        return jsonify({"error": "Input should be a list of camera configurations"}), 400

    response = []

    for config in data:
        if 'rtsp_link' not in config:
            return jsonify({"error": "RTSP link is required for all configurations"}), 400

        rtsp_link = config['rtsp_link']
        site_id = config['siteId']
        alarm_id = config['alarmId']
        camera_id = config['cameraId']
        display_width = config['display_width']
        display_height = config['display_height']
        line_coords = use_coordinates(config['line']['points'], config['line'])
        roi_coords = use_coordinates(config['roi']['points'], config['roi'])
        arrow_coords = config['arrow']  # Extract the arrow coordinates

        # Start a new process for each camera
        detection_process, stop_event = start_camera_process(rtsp_link, site_id,camera_id,alarm_id, roi_coords, line_coords,
                                                             arrow_coords, (display_width, display_height))
        detection_processes[camera_id] = {'process': detection_process, 'stop_event': stop_event}

        # # Add status for each camera to the response
        # response=json.dumps({
        #     "message": "Detection started",
        #     "camera_id": camera_id
        # })

    return jsonify({"message": "People counting started successfully for all cameras"}), 200


# Flask route to stop people counting detection for a specific camera
# Flask route to stop people counting detection for multiple cameras
@app.route('/zipline_stop', methods=['POST'])
def stop_counting():
    data = request.json.get('camera_ids', [])

    if not isinstance(data, list):
        return jsonify({"error": "Input should be a list of camera IDs"}), 400

    response = []

    for camera_id in data:
        if camera_id in detection_processes:
            detection_processes[camera_id]['stop_event'].set()
            detection_processes[camera_id]['process'].join()
            del detection_processes[camera_id]
            # response.append({
            #     "status": "Detection stopped",
            #     "camera_id": camera_id
            # })
        else:
            response.append({
                "error": "Camera not found",
                "camera_id": camera_id
            })

    return jsonify({"message": "People counting stoped successfully for all cameras"}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# # working in roi with detection of multiple cameras(not that much accurate)
# from flask import Flask, request, jsonify
# import numpy as np
# import cv2
# import math
# from ultralytics import YOLO
# from sort import Sort
# from multiprocessing import Process, Event
# import paho.mqtt.client as mqtt
# import json
#
# app = Flask(__name__)
#
# # Global variables
# mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
# mqtt_port = 1883
# mqtt_topic = "zipline/detected"
#
# # MQTT setup
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port)
# mqtt_client.loop_start()  # Non-blocking MQTT loop
#
# # MQTT publish function
# def publish_message(motion_type, rtsp_link, site_id, camera_id, alarm_id, people_count):
#     message = {
#         "rtsp_link": rtsp_link,
#         "siteId": site_id,
#         "cameraId": camera_id,
#         "alarmId":alarm_id,
#         "type": motion_type,
#         "people_count": people_count,
#     }
#     mqtt_client.publish(mqtt_topic, json.dumps(message))
#     print(f"Published message: {json.dumps(message)}")
#
#
# # Initialize tracker and model
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
# model = YOLO("../Model/yolov8l.pt")
#
# # Dictionary to store ongoing processes and stop events
# detection_processes = {}
#
# # Function to apply the given offsets to points in the JSON data
# def use_coordinates(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     placed_points = []
#     for point in points:
#         placed_x = int(point[0] + x_offset)
#         placed_y = int(point[1] + y_offset)
#         placed_points.append((placed_x, placed_y))
#
#     return placed_points
#
#
# # Check if the centroid is inside the ROI polygon
# def is_inside_roi(cx, cy, roi_points):
#     result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
#     return result >= 0
#
#
# # People counting logic based on ROI and zipline
# def detect_people_count(rtsp_url, site_id, camera_id, alarm_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
#     cap = cv2.VideoCapture(rtsp_url)
#     people_count = 0
#     prev_people_count = 0  # Keep track of the previous count
#     frame_skip = 2  # To process every 2nd frame
#     frame_count = 0
#
#     # Dictionary to store the previous Y position (cy) of each person to detect crossings
#     previous_positions = {}
#
#     display_width, display_height = display_size
#
#     # Arrow start and end points based on arrow coordinates
#     arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
#     arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))
#
#     # Create a window for each camera stream
#     window_name = f"Camera {camera_id}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     # Set the window size to the display size provided
#     cv2.resizeWindow(window_name, display_width, display_height)
#
#     # Error handling for the camera stream
#     if not cap.isOpened():
#         print(f"Error: Unable to open camera stream for camera {camera_id}")
#         return
#
#     while not stop_event.is_set():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Failed to read from camera {camera_id}")
#             break
#
#         # Resize the frame to match the provided display size
#         frame = cv2.resize(frame, (display_width, display_height))
#
#         # Draw zipline and ROI on the frame using the coordinates directly from the JSON
#         cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 3)  # Draw the zipline
#         cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)  # Draw the ROI
#
#         # Draw the arrow on the frame
#         cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)  # Draw the arrow
#
#         # Display the frame with the visual indicators (arrow, zipline, ROI)
#         cv2.imshow(window_name, frame)
#
#         frame_count += 1
#         if frame_count % frame_skip == 0:
#             # Create an ROI mask
#             roi_mask = np.zeros_like(frame[:, :, 0])  # Same size as one channel of the frame
#             cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)
#
#             # Apply the ROI mask to the frame
#             roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#
#             # Run YOLO model on the cropped frame (inside the ROI)
#             results = model(roi_frame, stream=True)
#             detections = np.empty((0, 5))
#
#             # Start tracking only when people are detected inside the ROI
#             tracking_started = False
#
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     x1b, y1b, x2b, y2b = box.xyxy[0]
#                     x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)
#
#                     # Confidence and class filtering (person class)
#                     conf = math.ceil((box.conf[0] * 100)) / 100
#                     cls = int(box.cls[0])
#                     if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
#                         detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
#                         tracking_started = True  # Begin tracking when a person is detected inside the ROI
#
#             # Proceed to tracking only if a person is detected inside the ROI
#             if tracking_started:
#                 tracked_people = tracker.update(detections)
#
#                 for person in tracked_people:
#                     x1, y1, x2, y2, person_id = person
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                     # Calculate the centroid of the bounding box
#                     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box
#
#                     # Check if the centroid is inside the ROI
#                     if not is_inside_roi(cx, cy, roi_coords):
#                         continue  # Skip if the person is outside the ROI
#
#                     # Calculate the y-coordinate of the zipline at the person's x-position (cx)
#                     m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
#                     b = zipline_coords[0][1] - m * zipline_coords[0][0]
#                     zipline_y = m * cx + b  # y-coordinate of the zipline at cx
#
#                     # Get the previous y-coordinate of the person (if any)
#                     prev_cy = previous_positions.get(person_id, cy)
#
#                     # Check if the person crossed the zipline (up or down)
#                     # If the arrow points up (arrow_end is above arrow_start), detect crossing upwards
#                     if arrow_coords['points'][1][1] < 0 and prev_cy > zipline_y and cy < zipline_y:
#                         people_count += 1
#                         publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, people_count)
#
#                     # If the arrow points down (arrow_end is below arrow_start), detect crossing downwards
#                     elif arrow_coords['points'][1][1] > 0 and prev_cy < zipline_y and cy > zipline_y:
#                         people_count += 1
#                         publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, people_count)
#
#                     # Update the previous position of the person
#                     previous_positions[person_id] = cy
#
#                     # Draw bounding box and centroid on the frame
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
#
#                 # Show people count on the frame
#                 cv2.putText(frame, f"Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#                 # Show the updated frame in the window for the camera
#                 cv2.imshow(window_name, frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_event.set()
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return people_count
#
#
#
# # Function to start each camera in its own process
# def start_camera_process(rtsp_url,site_id, camera_id,alarm_id,roi_coords, zipline_coords, arrow_coords, display_size):
#     stop_event = Event()
#     detection_process = Process(target=detect_people_count, args=(
#         rtsp_url,site_id, camera_id,alarm_id, roi_coords, zipline_coords, arrow_coords, display_size,stop_event))
#     detection_process.start()
#     return detection_process, stop_event
#
#
# # Flask route to start people counting detection for multiple cameras
# @app.route('/zipline_start', methods=['POST'])
# def start_counting():
#     data = request.json
#
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera configurations"}), 400
#
#     response = []
#
#     for config in data:
#         if 'rtsp_link' not in config:
#             return jsonify({"error": "RTSP link is required for all configurations"}), 400
#
#         rtsp_link = config['rtsp_link']
#         site_id = config['siteId']
#         alarm_id = config['alarmId']
#         camera_id = config['cameraId']
#         display_width = config['display_width']
#         display_height = config['display_height']
#         line_coords = use_coordinates(config['line']['points'], config['line'])
#         roi_coords = use_coordinates(config['roi']['points'], config['roi'])
#         arrow_coords = config['arrow']  # Extract the arrow coordinates
#
#         # Start a new process for each camera
#         detection_process, stop_event = start_camera_process(rtsp_link, site_id,camera_id,alarm_id, roi_coords, line_coords,
#                                                              arrow_coords, (display_width, display_height))
#         detection_processes[camera_id] = {'process': detection_process, 'stop_event': stop_event}
#
#         # # Add status for each camera to the response
#         # response=json.dumps({
#         #     "message": "Detection started",
#         #     "camera_id": camera_id
#         # })
#
#     return jsonify({"message": "People counting started successfully for all cameras"}), 200
#
#
# # Flask route to stop people counting detection for a specific camera
# # Flask route to stop people counting detection for multiple cameras
# @app.route('/zipline_stop', methods=['POST'])
# def stop_counting():
#     data = request.json.get('camera_ids', [])
#
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera IDs"}), 400
#
#     response = []
#
#     for camera_id in data:
#         if camera_id in detection_processes:
#             detection_processes[camera_id]['stop_event'].set()
#             detection_processes[camera_id]['process'].join()
#             del detection_processes[camera_id]
#             # response.append({
#             #     "status": "Detection stopped",
#             #     "camera_id": camera_id
#             # })
#         else:
#             response.append({
#                 "error": "Camera not found",
#                 "camera_id": camera_id
#             })
#
#     return jsonify({"message": "People counting stoped successfully for all cameras"}), 200
#
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# # count increase in specified roi
# from flask import Flask, request, jsonify
# import numpy as np
# import cv2
# import threading
# import math
# from ultralytics import YOLO
# from sort import Sort
#
# app = Flask(__name__)
#
# # Initialize tracker and model
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
# model = YOLO("../Model/yolov8l.pt")
#
# # Dictionary to store ongoing threads and stop events
# detection_threads = {}
#
# # Track the previous positions of people to check crossing direction
# previous_positions = {}
#
#
# # Function to apply the given offsets to points in the JSON data
# def use_coordinates(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     placed_points = []
#     for point in points:
#         placed_x = int(point[0] + x_offset)
#         placed_y = int(point[1] + y_offset)
#         placed_points.append((placed_x, placed_y))
#
#     return placed_points
#
#
# # Check if the centroid is inside the ROI polygon
# def is_inside_roi(cx, cy, roi_points):
#     result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
#     return result >= 0
#
#
# # People counting logic based on ROI and zipline
# def detect_people_count(rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
#     global previous_positions
#
#     cap = cv2.VideoCapture(rtsp_url)
#     people_count = 0
#     frame_skip = 2  # To process every 2nd frame
#     frame_count = 0
#
#     display_width, display_height = display_size
#
#     # Arrow start and end points based on arrow coordinates
#     arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
#     arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))
#
#     # Create a window for each camera stream
#     window_name = f"Camera {camera_id}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     # Set the window size to the display size provided
#     cv2.resizeWindow(window_name, display_width, display_height)
#
#     while not stop_event.is_set():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         # Resize the frame to match the provided display size
#         frame = cv2.resize(frame, (display_width, display_height))
#
#         frame_count += 1
#         if frame_count % frame_skip == 0:
#             # Create an ROI mask
#             roi_mask = np.zeros_like(frame[:, :, 0])  # Same size as one channel of the frame
#             cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)
#
#             # Apply the ROI mask to the frame
#             roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#
#             # Run YOLO model on the cropped frame (inside the ROI)
#             results = model(roi_frame, stream=True)
#             detections = np.empty((0, 5))
#
#             # Start tracking only when people are detected inside the ROI
#             tracking_started = False
#
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     x1b, y1b, x2b, y2b = box.xyxy[0]
#                     x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)
#
#                     # Confidence and class filtering (person class)
#                     conf = math.ceil((box.conf[0] * 100)) / 100
#                     cls = int(box.cls[0])
#                     if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
#                         detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
#                         tracking_started = True  # Begin tracking when a person is detected inside the ROI
#
#             # Proceed to tracking only if a person is detected inside the ROI
#             if tracking_started:
#                 tracked_people = tracker.update(detections)
#
#                 for person in tracked_people:
#                     x1, y1, x2, y2, person_id = person
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                     # Calculate the centroid of the bounding box
#                     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box
#
#                     # Check if the centroid is inside the ROI
#                     if not is_inside_roi(cx, cy, roi_coords):
#                         continue  # Skip if the person is outside the ROI
#
#                     # Check if the person crossed the zipline based on their current and previous positions
#                     m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
#                     b = zipline_coords[0][1] - m * zipline_coords[0][0]
#                     zipline_y = m * cx + b
#
#                     if person_id in previous_positions:
#                         prev_cy = previous_positions[person_id]
#
#                         # Check for "up" direction (from below the zipline to above it)
#                         if arrow_coords['points'][1][1] < 0 and prev_cy > zipline_y and cy < zipline_y:
#                             people_count += 1
#                         # Check for "down" direction (from above the zipline to below it)
#                         elif arrow_coords['points'][1][1] >= 0 and prev_cy < zipline_y and cy > zipline_y:
#                             people_count += 1
#
#                     # Update the person's previous position
#                     previous_positions[person_id] = cy
#
#                     # Draw bounding box and centroid on the frame
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
#
#                 # Draw zipline and ROI on the frame using the coordinates directly from the JSON
#                 cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 3)
#                 cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)
#
#                 # Draw the arrow on the frame
#                 cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)
#
#                 # Show people count on the frame
#                 cv2.putText(frame, f"Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#                 # Show the frame in the window for the camera
#                 cv2.imshow(window_name, frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_event.set()
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return people_count
#
#
# # Flask route to start people counting detection for multiple cameras
# @app.route('/start_counting', methods=['POST'])
# def start_counting():
#     data = request.json
#
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera configurations"}), 400
#
#     response = []
#
#     for config in data:
#         if 'rtsp_link' not in config:
#             return jsonify({"error": "RTSP link is required for all configurations"}), 400
#
#         rtsp_link = config['rtsp_link']
#         camera_id = config['cameraId']
#         display_width = config['display_width']
#         display_height = config['display_height']
#         line_coords = use_coordinates(config['line']['points'], config['line'])
#         roi_coords = use_coordinates(config['roi']['points'], config['roi'])
#         arrow_coords = config['arrow']  # Extract the arrow coordinates
#
#         # Start a new thread for each camera
#         stop_event = threading.Event()
#         detection_thread = threading.Thread(
#             target=detect_people_count,
#             args=(
#             rtsp_link, camera_id, roi_coords, line_coords, arrow_coords, (display_width, display_height), stop_event)
#         )
#         detection_threads[camera_id] = {'thread': detection_thread, 'stop_event': stop_event}
#         detection_thread.start()
#
#         # Add status for each camera to the response
#         response.append({
#             "status": "Detection started",
#             "camera_id": camera_id
#         })
#
#     return jsonify(response)
#
#
# # Flask route to stop people counting detection for a specific camera
# @app.route('/stop_counting/<int:camera_id>', methods=['POST'])
# def stop_counting(camera_id):
#     if camera_id in detection_threads:
#         detection_threads[camera_id]['stop_event'].set()
#         detection_threads[camera_id]['thread'].join()
#         del detection_threads[camera_id]
#         return jsonify({"status": "Detection stopped", "camera_id": camera_id})
#     else:
#         return jsonify({"error": "Camera not found"}), 404
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
#

# # working with multiple cameras in roi
# from flask import Flask, request, jsonify
# import numpy as np
# import cv2
# import math
# from ultralytics import YOLO
# from sort import Sort
# from multiprocessing import Process, Event
#
# app = Flask(__name__)
#
# # Initialize tracker and model
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
# model = YOLO("../Model/yolov8l.pt")
#
# # Dictionary to store ongoing processes and stop events
# detection_processes = {}
#
# # Track the previous positions of people to check crossing direction
# previous_positions = {}
#
#
# # Function to apply the given offsets to points in the JSON data
# def use_coordinates(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     placed_points = []
#     for point in points:
#         placed_x = int(point[0] + x_offset)
#         placed_y = int(point[1] + y_offset)
#         placed_points.append((placed_x, placed_y))
#
#     return placed_points
#
#
# # Check if the centroid is inside the ROI polygon
# def is_inside_roi(cx, cy, roi_points):
#     result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
#     return result >= 0
#
#
# # People counting logic based on ROI and zipline
# def detect_people_count(rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
#     global previous_positions
#
#     cap = cv2.VideoCapture(rtsp_url)
#     people_count = 0
#     frame_skip = 2  # To process every 2nd frame
#     frame_count = 0
#
#     display_width, display_height = display_size
#
#     # Arrow start and end points based on arrow coordinates
#     arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
#     arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))
#
#     # Create a window for each camera stream
#     window_name = f"Camera {camera_id}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     # Set the window size to the display size provided
#     cv2.resizeWindow(window_name, display_width, display_height)
#
#     while not stop_event.is_set():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         # Resize the frame to match the provided display size
#         frame = cv2.resize(frame, (display_width, display_height))
#
#         frame_count += 1
#         if frame_count % frame_skip == 0:
#             # Create an ROI mask
#             roi_mask = np.zeros_like(frame[:, :, 0])  # Same size as one channel of the frame
#             cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)
#
#             # Apply the ROI mask to the frame
#             roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#
#             # Run YOLO model on the cropped frame (inside the ROI)
#             results = model(roi_frame, stream=True)
#             detections = np.empty((0, 5))
#
#             # Start tracking only when people are detected inside the ROI
#             tracking_started = False
#
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     x1b, y1b, x2b, y2b = box.xyxy[0]
#                     x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)
#
#                     # Confidence and class filtering (person class)
#                     conf = math.ceil((box.conf[0] * 100)) / 100
#                     cls = int(box.cls[0])
#                     if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
#                         detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
#                         tracking_started = True  # Begin tracking when a person is detected inside the ROI
#
#             # Proceed to tracking only if a person is detected inside the ROI
#             if tracking_started:
#                 tracked_people = tracker.update(detections)
#
#                 for person in tracked_people:
#                     x1, y1, x2, y2, person_id = person
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                     # Calculate the centroid of the bounding box
#                     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box
#
#                     # Check if the centroid is inside the ROI
#                     if not is_inside_roi(cx, cy, roi_coords):
#                         continue  # Skip if the person is outside the ROI
#
#                     # Check if the person crossed the zipline based on their current and previous positions
#                     m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
#                     b = zipline_coords[0][1] - m * zipline_coords[0][0]
#                     zipline_y = m * cx + b
#
#                     if person_id in previous_positions:
#                         prev_cy = previous_positions[person_id]
#
#                         # Check for "up" direction (from below the zipline to above it)
#                         if arrow_coords['points'][1][1] < 0 and prev_cy > zipline_y and cy < zipline_y:
#                             people_count += 1
#                         # Check for "down" direction (from above the zipline to below it)
#                         elif arrow_coords['points'][1][1] >= 0 and prev_cy < zipline_y and cy > zipline_y:
#                             people_count += 1
#
#                     # Update the person's previous position
#                     previous_positions[person_id] = cy
#
#                     # Draw bounding box and centroid on the frame
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
#
#                 # Draw zipline and ROI on the frame using the coordinates directly from the JSON
#                 cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 3)
#                 cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)
#
#                 # Draw the arrow on the frame
#                 cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)
#
#                 # Show people count on the frame
#                 cv2.putText(frame, f"Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#                 # Show the frame in the window for the camera
#                 cv2.imshow(window_name, frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_event.set()
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return people_count
#
#
# # Function to start each camera in its own process
# def start_camera_process(rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size):
#     stop_event = Event()
#     detection_process = Process(target=detect_people_count, args=(
#         rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event))
#     detection_process.start()
#     return detection_process, stop_event
#
#
# # Flask route to start people counting detection for multiple cameras
# @app.route('/start_counting', methods=['POST'])
# def start_counting():
#     data = request.json
#
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera configurations"}), 400
#
#     response = []
#
#     for config in data:
#         if 'rtsp_link' not in config:
#             return jsonify({"error": "RTSP link is required for all configurations"}), 400
#
#         rtsp_link = config['rtsp_link']
#         camera_id = config['cameraId']
#         display_width = config['display_width']
#         display_height = config['display_height']
#         line_coords = use_coordinates(config['line']['points'], config['line'])
#         roi_coords = use_coordinates(config['roi']['points'], config['roi'])
#         arrow_coords = config['arrow']  # Extract the arrow coordinates
#
#         # Start a new process for each camera
#         detection_process, stop_event = start_camera_process(rtsp_link, camera_id, roi_coords, line_coords,
#                                                              arrow_coords, (display_width, display_height))
#         detection_processes[camera_id] = {'process': detection_process, 'stop_event': stop_event}
#
#         # Add status for each camera to the response
#         response.append({
#             "message": "Detection started",
#             "camera_id": camera_id
#         })
#
#     return jsonify(response)
#
#
# # Flask route to stop people counting detection for a specific camera
# @app.route('/stop_counting/<int:camera_id>', methods=['POST'])
# def stop_counting(camera_id):
#     if camera_id in detection_processes:
#         detection_processes[camera_id]['stop_event'].set()
#         detection_processes[camera_id]['process'].join()
#         del detection_processes[camera_id]
#         return jsonify({"status": "Detection stopped", "camera_id": camera_id})
#     else:
#         return jsonify({"error": "Camera not found"}), 404
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)





