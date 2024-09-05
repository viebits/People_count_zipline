# from flask import Flask, jsonify, request
# import cv2
# import numpy as np
# import paho.mqtt.client as mqtt
# import threading
# import time
# from ultralytics import YOLO
# import cvzone
# import math
# from sort import Sort
#
# app = Flask(__name__)
#
# # Initialize MQTT Client
# mqtt_client = mqtt.Client()
#
# # YOLO model initialization
# model = YOLO("../Model/yolov8l.pt")
#
# # Define class names for YOLO
# classNames = [
#     "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#     "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#     "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#     "teddy bear", "hair drier", "toothbrush"
# ]
#
# # Global variables
# cap = None
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)  # Initialize tracker
# frame_skip = 2  # Process every 2nd frame
# frame_count = 0
# tracker_results = []
#
# # Total counts
# totalCountUp = []
# totalCountDown = []
#
# crossedPersons = {}
# debounce_time = 1.5  # seconds
# last_count_time = {}
# buffer_vertical = 50  # Vertical buffer around the line
# buffer_horizontal = 50  # Horizontal buffer around the line
#
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code " + str(rc))
#
# def on_message(client, userdata, msg):
#     print(f"Message received on {msg.topic}: {str(msg.payload)}")
#
# mqtt_client.on_connect = on_connect
# mqtt_client.on_message = on_message
#
# # Connect to the MQTT broker
# mqtt_client.connect("192.168.1.120", 1883, 60)
#
# def set_roi_based_on_points(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     scaled_points = []
#     for point in points:
#         scaled_x = int(point[0] + x_offset)
#         scaled_y = int(point[1] + y_offset)
#         scaled_points.append((scaled_x, scaled_y))
#
#     return scaled_points
#
# def convert_line_points_to_absolute(line_data):
#     """
#     Converts line points from relative to absolute coordinates based on x, y, width, and height.
#     """
#     x_offset = line_data["x"]
#     y_offset = line_data["y"]
#     points = line_data["points"]
#
#     absolute_points = []
#     for point in points:
#         absolute_x = int(point[0] + x_offset)
#         absolute_y = int(point[1] + y_offset)
#         absolute_points.append((absolute_x, absolute_y))
#
#     return absolute_points
#
# def detect_motion(rtsp_url, camera_id, coordinates, line_data, motion_type, stop_event):
#     global roi_points, min_area, cap, line
#
#     cap = cv2.VideoCapture(rtsp_url)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#
#     # Initialize previous frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to read the stream.")
#         return
#
#     original_height, original_width = frame.shape[:2]
#     display_width = coordinates["display_width"]
#     display_height = coordinates["display_height"]
#
#     # Resize the frame
#     frame = cv2.resize(frame, (display_width, display_height))
#
#     # Set ROI based on API points
#     roi_points_from_api = coordinates["points"]
#     roi_points = set_roi_based_on_points(roi_points_from_api, coordinates)
#     roi_points = np.array(roi_points, dtype=np.int32)
#
#     # Create an ROI mask with the same size as the frame
#     roi_mask = np.zeros((display_height, display_width), dtype=np.uint8)
#     cv2.fillPoly(roi_mask, [roi_points], (255, 255, 255))
#
#     # Calculate the area of the ROI
#     roi_area = cv2.countNonZero(roi_mask)
#     full_frame_area = display_width * display_height
#     min_area = (1200 / full_frame_area) * roi_area  # Adjust the min area
#
#     # Convert line points to absolute coordinates
#     line = convert_line_points_to_absolute(line_data)
#
#     while not stop_event.is_set():
#         success, img = cap.read()
#         if not success:
#             break
#
#         # Resize the frame to match the display dimensions
#         img = cv2.resize(img, (display_width, display_height))
#
#         # Draw the ROI on the image
#         cv2.polylines(img, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)
#
#         # Draw the line on the image
#         cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 0), 3)
#
#         # Perform bitwise_and operation with properly sized ROI mask
#         imgRegion = cv2.bitwise_and(img, img, mask=roi_mask)
#
#         # Process only every nth frame
#         global frame_count, tracker_results
#         if frame_count % frame_skip == 0:
#             results = model(imgRegion, stream=True)
#
#             detections = np.empty((0, 5))
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     w, h = x2 - x1, y2 - y1
#
#                     conf = math.ceil((box.conf[0] * 100)) / 100
#                     cls = int(box.cls[0])
#                     currentClass = classNames[cls]
#
#                     if currentClass == "person" and conf > 0.3:
#                         currentArray = np.array([x1, y1, x2, y2, conf])
#                         detections = np.vstack((detections, currentArray))
#
#             tracker_results = tracker.update(detections)
#
#             # Update the line coordinates
#             m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])  # slope of the line
#             b = line[0][1] - m * line[0][0]  # y-intercept of the line
#
#             current_time = cv2.getTickCount() / cv2.getTickFrequency()
#
#             for result in tracker_results:
#                 x1, y1, x2, y2, id = result
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#                 cx, cy = x1 + w // 2, y1 + h // 2
#
#                 cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#                 cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
#                                    scale=2, thickness=3, offset=10)
#
#                 cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#                 if id not in crossedPersons:
#                     crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}
#
#                 previous_y = crossedPersons[id]['previous_y']
#                 line_y_at_cx = m * cx + b
#
#                 if (abs(cy - line_y_at_cx) <= buffer_vertical) and (line[0][0] - buffer_horizontal <= cx <= line[1][0] + buffer_horizontal):
#                     if id not in last_count_time or (current_time - last_count_time[id]) > debounce_time:
#                         if cy < line_y_at_cx and previous_y >= line_y_at_cx:
#                             if not crossedPersons[id]['crossed_up']:
#                                 totalCountUp.append(id)
#                                 crossedPersons[id]['crossed_up'] = True
#                                 crossedPersons[id]['crossed_down'] = False
#                                 cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 5)
#                                 last_count_time[id] = current_time
#
#                         elif cy > line_y_at_cx and previous_y <= line_y_at_cx:
#                             if not crossedPersons[id]['crossed_down']:
#                                 totalCountDown.append(id)
#                                 crossedPersons[id]['crossed_down'] = True
#                                 crossedPersons[id]['crossed_up'] = False
#                                 cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 5)
#                                 last_count_time[id] = current_time
#
#                 crossedPersons[id]['previous_y'] = cy
#
#         else:
#             # Draw tracker results on non-processing frames
#             for result in tracker_results:
#                 x1, y1, x2, y2, id = result
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#                 cx, cy = x1 + w // 2, y1 + h // 2
#
#                 cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#                 cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
#                                    scale=2, thickness=3, offset=10)
#
#                 cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#         # Display counts with labels
#         cv2.putText(img, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
#         cv2.putText(img, f'DOWN: {len(totalCountDown)}', (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#
#         # Show the frame in a window
#         cv2.imshow("Video Feed", img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             stop_event.set()
#             break
#
#         frame_count += 1
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# @app.route('/start_process', methods=['POST'])
# def start_process():
#     stop_event = threading.Event()
#
#     # Parse JSON input for coordinates and line points
#     data = request.json
#     camera_id = data["camera_id"]
#     rtsp_url = data["rtsp_url"]
#     coordinates = data["coordinates"]
#     line_data = data["line_data"]
#     motion_type = data.get("motion_type", "person")  # Default to 'person'
#
#     # Start the video feed processing in a new thread
#     threading.Thread(target=detect_motion, args=(rtsp_url, camera_id, coordinates, line_data, motion_type, stop_event)).start()
#
#     return jsonify({"status": "processing started"}), 200
#
# if __name__ == '__main__':
#     mqtt_client.loop_start()
#     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# import json
# import paho.mqtt.client as mqtt
# import numpy as np
# import cv2
# import cvzone
# import math
# from ultralytics import YOLO
# from sort import Sort
#
# app = Flask(__name__)
#
# # MQTT Settings
# broker_address = "192.168.1.120"  # MQTT Broker
# client = mqtt.Client("Flask_MQTT_Client")  # MQTT Client
#
# # Connect to the broker
# client.connect(broker_address)
#
# # Initialize the YOLO model
# model = YOLO("../Model/yolov8l.pt")
#
# # Initialize the SORT tracker
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
#
# # Class names for YOLO model
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]
#
# # Set ROI based on points and coordinates
# def set_roi_based_on_points(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     scaled_points = []
#     for point in points:
#         scaled_x = int(point[0] + x_offset)
#         scaled_y = int(point[1] + y_offset)
#         scaled_points.append((scaled_x, scaled_y))
#
#     return scaled_points
#
#
# @app.route('/process_json', methods=['POST'])
# def process_json():
#     # Get the JSON data from the request
#     data = request.get_json()
#
#     # Extract relevant information
#     rtsp_url = data[0]["rtsp_url"]
#     line_coordinates = data[0]["line"]
#     arrow_coordinates = data[0]["arrow"]
#     roi_coordinates = data[0]["roi"]
#
#     # Process line, arrow, and ROI coordinates using set_roi_based_on_points function
#     line_points = set_roi_based_on_points(line_coordinates["points"], line_coordinates)
#     arrow_points = set_roi_based_on_points(arrow_coordinates["points"], arrow_coordinates)
#     roi_points = set_roi_based_on_points(roi_coordinates["points"], roi_coordinates)
#
#     # Example: publishing the processed data to an MQTT topic
#     processed_data = {
#         "rtsp_url": rtsp_url,
#         "line_points": line_points,
#         "arrow_points": arrow_points,
#         "roi_points": roi_points
#     }
#     client.publish("camera/processed_coordinates", json.dumps(processed_data))
#
#     # Now we use the processed coordinates for further video processing (e.g., YOLO detection)
#     cap = cv2.VideoCapture(rtsp_url)
#
#     if not cap.isOpened():
#         return jsonify({"error": "Could not open RTSP stream"}), 500
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             print("Error: Could not read frame from the video stream.")
#             break
#
#         # Apply mask or ROI to the image
#         roi_mask = np.zeros_like(img, dtype=np.uint8)
#         roi_mask = cv2.fillPoly(roi_mask, [np.array(roi_points)], (255, 255, 255))
#         imgRegion = cv2.bitwise_and(img, roi_mask)
#
#         # Perform YOLO detection on the masked region
#         results = model(imgRegion, stream=True)
#         detections = np.empty((0, 5))
#
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 currentClass = classNames[cls]
#
#                 if currentClass == "person" and conf > 0.3:
#                     currentArray = np.array([x1, y1, x2, y2, conf])
#                     detections = np.vstack((detections, currentArray))
#
#         resultsTracker = tracker.update(detections)
#
#         # Draw the line and arrow based on processed points
#         cv2.line(img, tuple(line_points[0]), tuple(line_points[1]), (0, 0, 255), 3)
#         cv2.arrowedLine(img, tuple(arrow_points[0]), tuple(arrow_points[1]), (0, 255, 0), 3)
#
#         # Display results or publish them as needed
#         # For now, we can simply display the image or further publish MQTT messages with new detection data
#         # Assuming img is the frame you are displaying
#         cv2.namedWindow("YOLO Detection", cv2.WND_PROP_FULLSCREEN)
#         cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
#         cv2.imshow("YOLO Detection", img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return jsonify({"message": "Processed coordinates and performed YOLO detection", "data": processed_data}), 200
#
#
# # MQTT Callbacks
# def on_connect(client, userdata, flags, rc):
#     print(f"Connected to MQTT Broker with result code {rc}")
#     client.subscribe("camera/processed_coordinates")
#
#
# def on_message(client, userdata, msg):
#     print(f"Message received from {msg.topic}: {msg.payload.decode()}")
#
#
# client.on_connect = on_connect
# client.on_message = on_message
#
# if __name__ == '__main__':
#     client.loop_start()  # Start the MQTT client loop
#     app.run(host="0.0.0.0", port=5000, debug=True)


# working
# from flask import Flask, request, jsonify
# import json
# import paho.mqtt.client as mqtt
# import threading
# import numpy as np
# import cv2
# import cvzone
# import math
# from ultralytics import YOLO
# from sort import Sort
# import time
#
# app = Flask(__name__)
#
# # MQTT Settings
# broker_address = "192.168.1.120"  # MQTT Broker
# mqtt_client = mqtt.Client("Flask_MQTT_Client")  # MQTT Client
# mqtt_topic = "zipline/detected"
#
#
# # Connect to the broker
# mqtt_client.connect(broker_address)
#
# # Initialize the YOLO model
# model = YOLO("../Model/yolov8l.pt")
#
# # Initialize the SORT tracker
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
#
# # Class names for YOLO model
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]
#
# # Global variables for threading and synchronization
# frame_lock = threading.Lock()  # Thread lock for synchronizing access to the frame
# current_frame = None  # Stores the latest frame from the video stream
# rtsp_url = ""  # Stores the current RTSP URL from the JSON
# stop_threads = False  # Signal to stop threads when necessary
#
# # Global variables to track counts
# totalCountUp = []
# totalCountDown = []
# crossedPersons = {}
# arrow_direction = "up"  # Default direction based on the drawn arrow
#
#
# # MQTT publish function
# def publish_count(camera_id, people_count):
#     topic = f"{mqtt_topic}"
#     mqtt_client.publish(topic, str(people_count))
#
# def publish_message(motion_type, rtsp_link, site_id, camera_id, alarm_id, people_count):
#     message = {
#         "rtsp_link": rtsp_link,
#         "siteId": site_id,
#         "cameraId": camera_id,
#         "alarmId": alarm_id,
#         "type": motion_type
#
#     }
#     mqtt_client.publish(mqtt_topic, json.dumps(message))
#     print(f"Published message: {json.dumps(message)}")
#
#
#
# # Function to scale ROI based on points and coordinates
# def set_roi_based_on_points(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     scaled_points = []
#     for point in points:
#         scaled_x = int(point[0] + x_offset)
#         scaled_y = int(point[1] + y_offset)
#         scaled_points.append((scaled_x, scaled_y))
#
#     return scaled_points
#
# # Function to determine arrow direction
# def determine_arrow_direction(arrow_points):
#     # Calculate the direction based on the change in the y-values
#     x1, y1 = arrow_points[0]
#     x2, y2 = arrow_points[1]
#     # If the arrow goes upwards, we count 'up'. If it goes downwards, we count 'down'.
#     if y2 < y1:
#         return "up"
#     else:
#         return "down"
#
# # Function to draw the line, arrow, and ROI on the image (no masking, just highlighting)
# def draw_elements_on_image(img, line_points, arrow_points, roi_points):
#     # Draw the line
#     cv2.line(img, tuple(line_points[0]), tuple(line_points[1]), (0, 0, 255), 3)
#
#     # Draw the arrow
#     cv2.arrowedLine(img, tuple(arrow_points[0]), tuple(arrow_points[1]), (0, 255, 0), 3)
#
#     # Draw the ROI (Region of Interest) boundary without masking
#     cv2.polylines(img, [np.array(roi_points)], isClosed=True, color=(255, 255, 0), thickness=2)
#
# # Function to resize the frame and draw elements (without masking)
# def resize_frame_and_draw_elements(img, frame_width, frame_height, line_points, arrow_points, roi_points):
#     # Resize the frame to the target dimensions
#     resized_frame = cv2.resize(img, (frame_width, frame_height))
#
#     # Draw the line, arrow, and ROI on the resized image
#     draw_elements_on_image(resized_frame, line_points, arrow_points, roi_points)
#
#     return resized_frame
#
# # Background thread for processing video
# def video_processing(rtsp_url, line_points, arrow_points, roi_points, frame_width, frame_height):
#     global current_frame, stop_threads, totalCountUp, totalCountDown, crossedPersons, arrow_direction
#
#     # Determine the direction of the arrow based on its coordinates
#     arrow_direction = determine_arrow_direction(arrow_points)
#
#     cap = cv2.VideoCapture(rtsp_url)
#
#     if not cap.isOpened():
#         print("Error: Could not open RTSP stream")
#         return
#
#     # Define the debounce time and initial counting state
#     debounce_time = 1.5
#     last_count_time = {}
#
#     while not stop_threads:
#         success, img = cap.read()
#         if not success:
#             print("Error: Could not read frame from the video stream.")
#             break
#
#         # Resize the frame and draw elements (line, arrow, and ROI)
#         img_with_elements = resize_frame_and_draw_elements(img, frame_width, frame_height, line_points, arrow_points, roi_points)
#
#         # Now start detection after the visual elements are drawn
#         results = model(img_with_elements, stream=True)
#         detections = np.empty((0, 5))
#
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 currentClass = classNames[cls]
#
#                 if currentClass == "person" and conf > 0.3:
#                     currentArray = np.array([x1, y1, x2, y2, conf])
#                     detections = np.vstack((detections, currentArray))
#
#         resultsTracker = tracker.update(detections)
#
#         # Get the slope (m) and intercept (b) of the line to handle crossing logic
#         m = (line_points[1][1] - line_points[0][1]) / (line_points[1][0] - line_points[0][0])
#         b = line_points[0][1] - m * line_points[0][0]
#
#         current_time = cv2.getTickCount() / cv2.getTickFrequency()
#
#         # Counting logic after drawing
#         for result in resultsTracker:
#             x1, y1, x2, y2, id = result
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             cx, cy = x1 + w // 2, y1 + h // 2
#
#             # Draw rectangle and ID around the detected person
#             cvzone.cornerRect(img_with_elements, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#             cvzone.putTextRect(img_with_elements, f' {int(id)}', (max(0, x1), max(35, y1)),
#                                scale=2, thickness=3, offset=10)
#
#             cv2.circle(img_with_elements, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#             if id not in crossedPersons:
#                 crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}
#
#             previous_y = crossedPersons[id]['previous_y']
#
#             # Calculate the y-value on the line at the object's x-coordinate (cx)
#             line_y_at_cx = m * cx + b
#
#             if (cy < line_y_at_cx and previous_y >= line_y_at_cx) and not crossedPersons[id]['crossed_up']:
#                 totalCountUp.append(id)
#                 crossedPersons[id]['crossed_up'] = True
#                 crossedPersons[id]['crossed_down'] = False
#                 last_count_time[id] = current_time
#
#             elif (cy > line_y_at_cx and previous_y <= line_y_at_cx) and not crossedPersons[id]['crossed_down']:
#                 totalCountDown.append(id)
#                 crossedPersons[id]['crossed_down'] = True
#                 crossedPersons[id]['crossed_up'] = False
#                 last_count_time[id] = current_time
#
#             crossedPersons[id]['previous_y'] = cy
#
#         # Display counts based on the arrow direction (only show "up" or "down" count based on the arrow)
#         if arrow_direction == "up":
#             cv2.putText(img_with_elements, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
#         elif arrow_direction == "down":
#             cv2.putText(img_with_elements, f'DOWN: {len(totalCountDown)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#
#         # Show final frame with detections and counts
#         with frame_lock:
#             current_frame = img_with_elements
#
#         # Show the processed frame either in full-screen or resized based on the given dimensions
#         if frame_width > 0 and frame_height > 0:
#             cv2.imshow("YOLO Detection", img_with_elements)
#         else:
#             cv2.namedWindow("YOLO Detection", cv2.WND_PROP_FULLSCREEN)
#             cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#             cv2.imshow("YOLO Detection", img_with_elements)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             stop_threads = True
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Flask route to handle incoming JSON and start processing
# @app.route('/zipline_counting', methods=['POST'])
# def process_json():
#     global rtsp_url, stop_threads
#
#     # Stop previous threads if necessary
#     stop_threads = True
#     time.sleep(1)  # Ensure threads have stopped
#
#     # Reset stop signal for new threads
#     stop_threads = False
#
#     # Get the JSON data from the request
#     data = request.get_json()
#
#     # Extract relevant information
#     rtsp_url = data[0]["rtsp_url"]
#     line_coordinates = data[0]["line"]
#     arrow_coordinates = data[0]["arrow"]
#     roi_coordinates = data[0]["roi"]
#     frame_width = data[0].get("frame_width", 0)
#     frame_height = data[0].get("frame_height", 0)
#
#     # Process line, arrow, and ROI coordinates using set_roi_based_on_points function
#     line_points = set_roi_based_on_points(line_coordinates["points"], line_coordinates)
#     arrow_points = set_roi_based_on_points(arrow_coordinates["points"], arrow_coordinates)
#     roi_points = set_roi_based_on_points(roi_coordinates["points"], roi_coordinates)
#
#     # Start video processing in a new thread
#     processing_thread = threading.Thread(target=video_processing, args=(rtsp_url, line_points, arrow_points, roi_points, frame_width, frame_height))
#     processing_thread.start()
#
#     return jsonify({"message": "Processing started", "rtsp_url": rtsp_url}), 200
#
# # MQTT Callbacks
# def on_connect(client, userdata, flags, rc):
#     print(f"Connected to MQTT Broker with result code {rc}")
#     client.subscribe("camera/processed_coordinates")
#
# def on_message(client, userdata, msg):
#     print(f"Message received from {msg.topic}: {msg.payload.decode()}")
#
# mqtt_client.on_connect = on_connect
# mqtt_client.on_message = on_message
#
# # Start the Flask app and MQTT client loop
# if __name__ == '__main__':
#     mqtt_client.loop_start()  # Start the MQTT client loop
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import math
from ultralytics import YOLO
from sort import Sort
import multiprocessing
import paho.mqtt.client as mqtt

app = Flask(__name__)

# MQTT Settings
broker_address = "192.168.1.120"
mqtt_client = mqtt.Client("Flask_MQTT_Client")
mqtt_topic = "zipline/detected"
mqtt_client.connect(broker_address)

# Dictionary to store active processes
active_processes = {}

# Class names for YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# MQTT publish function
def publish_message(cameraId, motion_type, people_count, rtsp_link, siteId, alarmId):
    message = {
        "cameraId": cameraId,
        "type": motion_type,
        "peopleCount": people_count,
        "rtsp_link": rtsp_link,
        "siteId": siteId,
        "alarmId": alarmId
    }
    mqtt_client.publish(mqtt_topic, json.dumps(message))
    print(f"Published message for camera {cameraId}: {json.dumps(message)}")

# Function to set ROI based on points and coordinates
def set_roi_based_on_points(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points

# Function to determine arrow direction
def determine_arrow_direction(arrow_points):
    x1, y1 = arrow_points[0]
    x2, y2 = arrow_points[1]
    return "up" if y2 < y1 else "down"

# Function to draw elements on the image
def draw_elements_on_image(img, line_points, arrow_points, roi_points):
    cv2.line(img, tuple(line_points[0]), tuple(line_points[1]), (0, 0, 255), 3)
    cv2.arrowedLine(img, tuple(arrow_points[0]), tuple(arrow_points[1]), (0, 255, 0), 3)
    cv2.polylines(img, [np.array(roi_points)], isClosed=True, color=(255, 255, 0), thickness=2)

# Function to resize the frame and draw elements
def resize_frame_and_draw_elements(img, display_width, display_height, line_points, arrow_points, roi_points):
    resized_frame = cv2.resize(img, (display_width, display_height))
    draw_elements_on_image(resized_frame, line_points, arrow_points, roi_points)
    return resized_frame

# Video processing function (runs in each process)
def video_processing(camera_data):
    rtsp_link = camera_data["rtsp_link"]
    cameraId = camera_data["cameraId"]
    line_points = set_roi_based_on_points(camera_data["line"]["points"], camera_data["line"])
    arrow_points = set_roi_based_on_points(camera_data["arrow"]["points"], camera_data["arrow"])
    roi_points = set_roi_based_on_points(camera_data["roi"]["points"], camera_data["roi"])
    display_width = camera_data["display_width"]
    display_height = camera_data["display_height"]
    siteId = camera_data["siteId"]
    alarmId = camera_data["alarmId"]

    # Initialize YOLO and SORT for each process
    model = YOLO("../Model/yolov8l.pt")  # YOLO model must be initialized in the process
    tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)  # SORT tracker initialized in the process

    totalCountUp = []
    totalCountDown = []
    crossedPersons = {}

    arrow_direction = determine_arrow_direction(arrow_points)

    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for camera {cameraId}")
        return

    while True:
        success, img = cap.read()
        if not success:
            print(f"Error: Could not read frame from camera {cameraId}")
            break

        # Resize the frame and draw elements (line, arrow, and ROI)
        img_with_elements = resize_frame_and_draw_elements(img, display_width, display_height, line_points, arrow_points, roi_points)

        # Object detection
        results = model(img_with_elements, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        # Line crossing logic
        m = (line_points[1][1] - line_points[0][1]) / (line_points[1][0] - line_points[0][0])
        b = line_points[0][1] - m * line_points[0][0]

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if id not in crossedPersons:
                crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}

            previous_y = crossedPersons[id]['previous_y']
            line_y_at_cx = m * cx + b

            if (cy < line_y_at_cx and previous_y >= line_y_at_cx) and not crossedPersons[id]['crossed_up']:
                totalCountUp.append(id)
                crossedPersons[id]['crossed_up'] = True
                crossedPersons[id]['crossed_down'] = False

            elif (cy > line_y_at_cx and previous_y <= line_y_at_cx) and not crossedPersons[id]['crossed_down']:
                totalCountDown.append(id)
                crossedPersons[id]['crossed_down'] = True
                crossedPersons[id]['crossed_up'] = False

            crossedPersons[id]['previous_y'] = cy

        # Publish MQTT message
        people_count = len(totalCountUp) if arrow_direction == "up" else len(totalCountDown)
        publish_message(cameraId, arrow_direction, people_count, rtsp_link, siteId, alarmId)

        # Show the frame with counts
        if arrow_direction == "up":
            cv2.putText(img_with_elements, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        elif arrow_direction == "down":
            cv2.putText(img_with_elements, f'DOWN: {len(totalCountDown)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow(f"Camera {cameraId}", img_with_elements)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start video processing for cameras
@app.route('/zipline_counting', methods=['POST'])
def start_cameras():
    data = request.get_json()

    for camera_data in data:
        camera_id = camera_data['cameraId']
        if camera_id in active_processes and active_processes[camera_id].is_alive():
            print(f"Camera {camera_id} is already being processed.")
        else:
            process = multiprocessing.Process(target=video_processing, args=(camera_data,))
            process.start()
            active_processes[camera_id] = process
            print(f"Started video processing for camera {camera_id}")

    return jsonify({"message": "Started processing for cameras."}), 200

# Stop video processing for cameras
@app.route('/stop_cameras', methods=['POST'])
def stop_cameras():
    data = request.get_json()
    camera_ids = data['camera_ids']

    for camera_id in camera_ids:
        process = active_processes.get(camera_id)
        if process and process.is_alive():
            print(f"Stopping process for camera {camera_id}")
            process.terminate()  # Terminate the process
            process.join()  # Ensure process has finished
            del active_processes[camera_id]
            print(f"Stopped process for camera ID {camera_id}.")
        else:
            print(f"No active process found for camera ID {camera_id}.")

    return jsonify({"message": f"Stopped processes for camera IDs: {camera_ids}"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
