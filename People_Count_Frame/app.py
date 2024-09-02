# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
# from flask import Flask, request, jsonify
# import paho.mqtt.client as mqtt
#
# app = Flask(__name__)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     exit()
#
# # Default frame dimensions (width, height)
# frame_width = 1000
# frame_height = 800
#
# # Frame storage and ROI coordinates
# frame = None
# roi_top_left = (200, 200)
# roi_bottom_right = (600, 500)
#
# # MQTT Configuration
# mqtt_broker = "192.168.1.120"  # Example broker; change to your broker
# mqtt_port = 1883
# mqtt_topic = "yolov8/people_count"
#
# # Initialize MQTT client
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port, 60)
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
# # Flask route to set the frame dimensions and ROI coordinates
# @app.route('/set_config', methods=['POST'])
# def set_config():
#     global frame_width, frame_height, roi_top_left, roi_bottom_right
#     data = request.json
#
#     # Update frame dimensions
#     if 'frame_width' in data and 'frame_height' in data:
#         frame_width = data['frame_width']
#         frame_height = data['frame_height']
#
#     # Update ROI coordinates
#     if 'top_left' in data and 'bottom_right' in data:
#         roi_top_left = tuple(data['top_left'])
#         roi_bottom_right = tuple(data['bottom_right'])
#
#     return jsonify({
#         "message": "Configuration updated",
#         "frame_width": frame_width,
#         "frame_height": frame_height,
#         "roi_top_left": roi_top_left,
#         "roi_bottom_right": roi_bottom_right
#     }), 200
#
# # Flask route to get the current people count
# @app.route('/get_people_count', methods=['GET'])
# def get_people_count():
#     if frame is None:
#         return jsonify({"error": "Frame not available"}), 500
#
#     # Resize the frame
#     resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Create a transparent ROI overlay
#     overlay = resized_frame.copy()
#     alpha = 0.3  # Transparency factor
#     cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle
#     cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0, resized_frame)
#
#     # Create an ROI mask
#     roi_mask = cv2.rectangle(
#         np.zeros_like(resized_frame, dtype=np.uint8),
#         roi_top_left,
#         roi_bottom_right,
#         (255, 255, 255),
#         thickness=-1
#     )
#
#     # Apply the ROI mask to the frame using bitwise AND
#     masked_frame = cv2.bitwise_and(resized_frame, roi_mask)
#
#     # Run YOLOv8 inference on the masked frame
#     results = model(masked_frame, conf=0.3, iou=0.4)
#
#     # Initialize people count
#     people_count = 0
#
#     # Iterate through detected objects
#     for box in results[0].boxes.data:
#         class_id = int(box[5])
#         if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
#             people_count += 1
#
#     # Publish the people count to the MQTT broker
#     mqtt_client.publish(mqtt_topic, str(people_count))
#
#     return jsonify({"people_count": people_count})
#
# # Flask route to start the detection
# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     return jsonify({"message": "Detection started"}), 200
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
#     cap.release()
#     cv2.destroyAllWindows()


# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
# from flask import Flask, request, jsonify
# import paho.mqtt.client as mqtt
#
# app = Flask(__name__)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     exit()
#
# # Default frame dimensions (width, height)
# frame_width = 1000
# frame_height = 800
#
# # Frame storage and ROI coordinates
# frame = None
# roi_top_left = (200, 200)
# roi_bottom_right = (600, 500)
#
# # MQTT Configuration
# mqtt_broker = "192.168.1.120"  # Example broker; change to your broker
# mqtt_port = 1883
# mqtt_topic = "yolov8/people_count"
#
# # Initialize MQTT client
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port, 60)
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
# # Unified Flask route for configuration and detection
# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     global frame_width, frame_height, roi_top_left, roi_bottom_right
#     data = request.json
#
#     # Update frame dimensions
#     if 'frame_width' in data and 'frame_height' in data:
#         frame_width = data['frame_width']
#         frame_height = data['frame_height']
#
#     # Update ROI coordinates
#     if 'top_left' in data and 'bottom_right' in data:
#         roi_top_left = tuple(data['top_left'])
#         roi_bottom_right = tuple(data['bottom_right'])
#
#     # Wait for a frame to be captured
#     if frame is None:
#         return jsonify({"error": "Frame not available"}), 500
#
#     # Resize the frame
#     resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Create a transparent ROI overlay
#     overlay = resized_frame.copy()
#     alpha = 0.3  # Transparency factor
#     cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle
#     cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0, resized_frame)
#
#     # Create an ROI mask
#     roi_mask = cv2.rectangle(
#         np.zeros_like(resized_frame, dtype=np.uint8),
#         roi_top_left,
#         roi_bottom_right,
#         (255, 255, 255),
#         thickness=-1
#     )
#
#     # Apply the ROI mask to the frame using bitwise AND
#     masked_frame = cv2.bitwise_and(resized_frame, roi_mask)
#
#     # Run YOLOv8 inference on the masked frame
#     results = model(masked_frame, conf=0.3, iou=0.4)
#
#     # Initialize people count
#     people_count = 0
#
#     # Iterate through detected objects
#     for box in results[0].boxes.data:
#         class_id = int(box[5])
#         if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
#             people_count += 1
#
#     # Publish the people count to the MQTT broker
#     mqtt_client.publish(mqtt_topic, str(people_count))
#
#     return jsonify({
#         "message": "Detection complete",
#         "people_count": people_count,
#         "frame_width": frame_width,
#         "frame_height": frame_height,
#         "roi_top_left": roi_top_left,
#         "roi_bottom_right": roi_bottom_right
#     })
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
#     cap.release()
#     cv2.destroyAllWindows()


#working
# import cv2
# import threading
# import numpy as np
# import json
# import argparse
# from ultralytics import YOLO
# import paho.mqtt.client as mqtt
# import sys
#
# # Argument parser for command-line inputs
# parser = argparse.ArgumentParser(description="Process frame with YOLOv8")
# parser.add_argument('--input_json', type=str, required=True, help='Path to JSON input file')
#
# args = parser.parse_args()
#
# # Load the JSON input from the file
# with open(args.input_json, 'r') as f:
#     input_data = json.load(f)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # RTSP stream URL
# rtsp_url = input_data.get("rtsp_url", 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live')
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     sys.exit(1)
#
# # Default frame dimensions (width, height) from JSON input
# frame_width = input_data.get("frame_width", 1000)
# frame_height = input_data.get("frame_height", 800)
#
# # ROI coordinates from JSON input
# roi_top_left = tuple(input_data.get("top_left", [200, 200]))
# roi_bottom_right = tuple(input_data.get("bottom_right", [600, 500]))
#
# # MQTT Configuration
# mqtt_broker = input_data.get("mqtt_broker", "192.168.1.120")
# mqtt_port = input_data.get("mqtt_port", 1883)
# mqtt_topic = input_data.get("mqtt_topic", "yolov8/people_count")
#
# # Initialize MQTT client
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port, 60)
#
# # Combined function to capture and process frames
# def capture_and_process_frames():
#     global frame
#
#     while True:
#         # Capture a frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#         print("Frame captured successfully")
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # Create a transparent ROI overlay
#         overlay = resized_frame.copy()
#         alpha = 0.3  # Transparency factor
#         cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle
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
#
#         # Publish the people count to the MQTT broker
#         mqtt_client.publish(mqtt_topic, str(people_count))
#
#         # Output the result
#         print(json.dumps({
#             "message": "Detection complete",
#             "people_count": people_count,
#             "frame_width": frame_width,
#             "frame_height": frame_height,
#             "roi_top_left": roi_top_left,
#             "roi_bottom_right": roi_bottom_right
#         }, indent=4))
#
# if __name__ == '__main__':
#     capture_and_process_frames()
#     cap.release()
#     cv2.destroyAllWindows()

# import cv2
# import threading
# import numpy as np
# import json
# import argparse
# from ultralytics import YOLO
# import paho.mqtt.client as mqtt
# import sys
#
# # Argument parser for command-line inputs
# parser = argparse.ArgumentParser(description="Process frame with YOLOv8")
# parser.add_argument('--input_json', type=str, required=True, help='Path to JSON input file')
#
# args = parser.parse_args()
#
# # Load the JSON input from the file
# with open(args.input_json, 'r') as f:
#     input_data = json.load(f)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # RTSP stream URL
# rtsp_url = input_data.get("rtsp_url", 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live')
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     sys.exit(1)
#
# # Default frame dimensions (width, height) from JSON input
# frame_width = input_data.get("frame_width", 1400)
# frame_height = input_data.get("frame_height", 700)
#
# # ROI coordinates from JSON input
# roi_top_left = tuple(input_data.get("top_left", [200, 200]))
# roi_bottom_right = tuple(input_data.get("bottom_right", [600, 500]))
#
# # MQTT Configuration
# mqtt_broker = input_data.get("mqtt_broker", "192.168.1.120")
# mqtt_port = input_data.get("mqtt_port", 1883)
# mqtt_topic = input_data.get("mqtt_topic", "yolov8/people_count")
#
# # Initialize MQTT client
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port, 60)
#
# # Combined function to capture, process, and display frames
# def capture_process_and_display_frames():
#     global frame
#
#     while True:
#         # Capture a frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#         print("Frame captured successfully")
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # Create a transparent ROI overlay
#         overlay = resized_frame.copy()
#         alpha = 0.3  # Transparency factor
#         cv2.rectangle(overlay, roi_top_left, roi_bottom_right, (0, 255, 0), -1)  # Green filled rectangle
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
#                 # Draw bounding box on the frame
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#                 cv2.putText(resized_frame, f'Person {people_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
#
#         # Publish the people count to the MQTT broker
#         mqtt_client.publish(mqtt_topic, str(people_count))
#
#         # Display the frame with bounding boxes
#         cv2.imshow('YOLOv8 People Detection', resized_frame)
#
#         # Check for 'q' key to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting...")
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     capture_process_and_display_frames()


# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
# import paho.mqtt.client as mqtt
# import time
# import json
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: Could not open RTSP stream.")
#     exit()
#
# # Desired frame dimensions (width, height)
# frame_width = 1400
# frame_height = 700
#
# # Global variables
# frame = None
# people_count = 0
# points = []
# roi_defined = False
# roi_mask = None
# processing = False
#
# # MQTT setup
# mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
# mqtt_port = 1883
# mqtt_topic = "people_count"
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port)
#
#
# def publish_count():
#     global people_count
#     while processing:
#         mqtt_client.publish(mqtt_topic, str(people_count))
#         time.sleep(5)  # Publish every 5 seconds
#
#
# # Frame capture and processing function
# def capture_and_process_frames():
#     global frame, people_count, roi_mask, processing
#     while processing:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#
#         # Resize the frame
#         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#
#         # Create a copy of the frame for display
#         display_frame = resized_frame.copy()
#
#         # Draw the ROI on the display frame
#         if points:
#             cv2.polylines(display_frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
#
#         # Ensure roi_mask is not None and has the same shape as resized_frame
#         if roi_mask is not None and roi_mask.shape[:2] == resized_frame.shape[:2]:
#             # Apply the ROI mask to the frame using bitwise AND
#             imgRegion = cv2.bitwise_and(resized_frame, roi_mask)
#
#             # Run YOLOv8 inference on the masked frame
#             results = model(imgRegion, conf=0.3, iou=0.4)
#
#             # Count people
#             people_count = sum(1 for box in results[0].boxes.data if int(box[5]) == 0)
#
#             # Draw bounding boxes for detected people
#             for box in results[0].boxes.data:
#                 if int(box[5]) == 0:  # Class 0 is typically 'person'
#                     x1, y1, x2, y2 = map(int, box[:4])
#                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#         # Display the count on the frame
#         cv2.putText(display_frame, f"People count: {people_count}", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Show the frame
#         cv2.imshow('People Count', display_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             processing = False
#             break
#
#     cv2.destroyAllWindows()
#
#
# def start_counting_from_file(filename):
#     global points, roi_defined, roi_mask, processing
#
#     # Read JSON data from the file
#     try:
#         with open(filename, 'r') as file:
#             data = json.load(file)
#     except FileNotFoundError:
#         print(f"Error: File '{filename}' not found.")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON in file '{filename}'.")
#         return
#
#     if 'coordinates' not in data:
#         print("Error: Missing 'coordinates' in JSON data")
#         return
#
#     points = data['coordinates']
#
#     if len(points) < 3:
#         print("Error: At least 3 points are required to define an ROI")
#         return
#
#     roi_defined = True
#
#     # Create an ROI mask based on the selected free-form ROI
#     roi_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
#     cv2.fillPoly(roi_mask, [np.array(points)], (255, 255, 255))
#
#     # Start processing if not already started
#     if not processing:
#         processing = True
#
#         # Start the frame capture and processing thread
#         processing_thread = threading.Thread(target=capture_and_process_frames)
#         processing_thread.start()
#
#         # Start the MQTT publishing thread
#         mqtt_thread = threading.Thread(target=publish_count)
#         mqtt_thread.start()
#
#     print("People counting started successfully")
#
#
# if __name__ == '__main__':
#     json_filename = 'input.json'  # Replace with your JSON file name
#     start_counting_from_file(json_filename)
#
#     # Keep the main thread running
#     try:
#         while processing:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("Stopping the application...")
#         processing = False
#

#working but aking the coordinates directly not compared to width and height
# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
# from flask import Flask, jsonify, request
# import paho.mqtt.client as mqtt
# import time
#
# app = Flask(__name__)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
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
# frame_width = 1400
# frame_height = 700
#
# # Global variables
# frame = None
# people_count = 0
# points = []
# roi_defined = False
# roi_mask = None
# processing = False
#
# # MQTT setup
# mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
# mqtt_port = 1883
# mqtt_topic = "people_count"
#
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port)
#
#
# def publish_count():
#     global people_count
#     while processing:
#         mqtt_client.publish(mqtt_topic, str(people_count))
#         time.sleep(5)  # Publish every 5 seconds
#
#
# # Frame capture and processing thread
# def capture_and_process_frames():
#     global frame, people_count, roi_mask, processing
#     while processing:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
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
#         # Count people
#         people_count = sum(1 for box in results[0].boxes.data if int(box[5]) == 0)
#
#         # Draw the ROI on the frame
#         frame_with_roi = resized_frame.copy()
#         cv2.polylines(frame_with_roi, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
#
#         # Display the count on the frame
#         cv2.putText(frame_with_roi, f"People count: {people_count}", (30, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Show the frame
#         cv2.imshow('People Count', frame_with_roi)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             processing = False
#             break
#
#     cv2.destroyAllWindows()
#
#
# @app.route('/start_counting', methods=['POST'])
# def start_counting():
#     global points, roi_defined, roi_mask, processing
#
#     # Get JSON data from the request
#     data = request.json
#
#     if 'coordinates' not in data:
#         return jsonify({"error": "Missing 'coordinates' in JSON data"}), 400
#
#     points = data['coordinates']
#
#     if len(points) < 3:
#         return jsonify({"error": "At least 3 points are required to define an ROI"}), 400
#
#     roi_defined = True
#
#     # Create an ROI mask based on the selected free-form ROI
#     roi_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
#     cv2.fillPoly(roi_mask, [np.array(points)], (255, 255, 255))
#
#     # Start processing if not already started
#     if not processing:
#         processing = True
#
#         # Start the frame capture and processing thread
#         processing_thread = threading.Thread(target=capture_and_process_frames, daemon=True)
#         processing_thread.start()
#
#         # Start the MQTT publishing thread
#         mqtt_thread = threading.Thread(target=publish_count, daemon=True)
#         mqtt_thread.start()
#
#     return jsonify({"message": "People counting started successfully"}), 200
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# import cv2
# import threading
# import numpy as np
# import json
# from ultralytics import YOLO
# from flask import Flask, jsonify, request
# import paho.mqtt.client as mqtt
# import time
#
# app = Flask(__name__)
#
# # Load the YOLOv8 model
# model = YOLO('../Model/yolov8l.pt')
#
# # Global variables
# frame = None
# roi_points = []
# roi_defined = False
# processing = False
# mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
# mqtt_port = 1883
# mqtt_topic = "people_count"
# cap = None
#
# # MQTT setup
# mqtt_client = mqtt.Client()
# mqtt_client.connect(mqtt_broker, mqtt_port)
# mqtt_client.loop_start()  # Start the MQTT loop
#
#
# # MQTT publish function
# def publish_count(people_count):
#     mqtt_client.publish(mqtt_topic, str(people_count))
#
#
# # Function to adjust ROI points based on provided coordinates
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
# # Function to capture frames from the RTSP stream
# def capture_frames(rtsp_url):
#     global frame, cap
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print("Error: Could not open RTSP stream.")
#         return
#
#     while processing:
#         ret, captured_frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from RTSP stream.")
#             break
#         frame = captured_frame
#
#
# # Function to process frames and perform detection
# def process_frames(coordinates):
#     global frame, roi_points, roi_defined, processing
#
#     # Adjust ROI points based on the provided coordinates
#     roi_points = set_roi_based_on_points(coordinates["points"], coordinates)
#     roi_defined = True
#
#     # Create the ROI mask
#     roi_mask = np.zeros((coordinates["display_height"], coordinates["display_width"]), dtype=np.uint8)
#     cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], (255, 255, 255))
#
#     while processing:
#         if frame is not None:
#             # Resize the frame to the display size
#             resized_frame = cv2.resize(frame, (coordinates["display_width"], coordinates["display_height"]))
#
#             # Apply the ROI mask to the frame using bitwise AND
#             imgRegion = cv2.bitwise_and(resized_frame, resized_frame, mask=roi_mask)
#
#             # Run YOLOv8 inference on the masked frame
#             results = model(imgRegion, conf=0.3, iou=0.4)
#
#             # Initialize people count
#             people_count = 0
#
#             # Iterate through detected objects
#             for box in results[0].boxes.data:
#                 class_id = int(box[5])
#                 if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
#                     people_count += 1
#                     # Draw bounding box
#                     x1, y1, x2, y2 = map(int, box[:4])
#                     cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
#                     # Add label
#                     cv2.putText(resized_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0),
#                                 1)
#
#             # Display the number of people detected on the frame
#             cv2.putText(resized_frame, f"People count: {people_count}", (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # Draw the selected ROI on the live stream
#             if len(roi_points) > 1:
#                 cv2.polylines(resized_frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)
#
#             # Show the annotated frame
#             cv2.imshow('People Count', resized_frame)
#
#             # Publish the count to MQTT
#             publish_count(people_count)
#
#             # Break the loop on 'q' key press or window close
#             if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
#                 processing = False
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # Flask route to start people counting
# @app.route('/start_counting', methods=['POST'])
# def start_counting():
#     global processing, cap
#
#     # Get JSON data from the request
#     data = request.json
#
#     # Validate the input data
#     required_keys = ['rtsp_url', 'x', 'y', 'frame_width', 'frame_height', 'points']
#     if not all(key in data for key in required_keys):
#         return jsonify({"error": "Missing one or more required fields in JSON data"}), 400
#
#     # Extract the values
#     rtsp_url = data['rtsp_url']
#     x = data['x']
#     y = data['y']
#     display_width = data['frame_width']
#     display_height = data['frame_height']
#     points = data['points']
#
#     # Validate points
#     if not isinstance(points, list) or len(points) < 3:
#         return jsonify({"error": "At least 3 points are required to define an ROI"}), 400
#
#     # Combine the extracted values into a coordinates dictionary
#     coordinates = {
#         "x": x,
#         "y": y,
#         "display_width": display_width,
#         "display_height": display_height,
#         "points": points
#     }
#
#     if cap is not None:
#         cap.release()
#
#     # Start the frame capture in a separate thread
#     processing = True
#     capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True)
#     capture_thread.start()
#
#     # Start the detection process in a separate thread
#     processing_thread = threading.Thread(target=process_frames, args=(coordinates,), daemon=True)
#     processing_thread.start()
#
#     return jsonify({"message": "People counting started successfully"}), 200
#
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


import cv2
import multiprocessing
import numpy as np
import json
from ultralytics import YOLO
from flask import Flask, jsonify, request
import paho.mqtt.client as mqtt

app = Flask(__name__)

# Load the YOLOv8 model globally so each process can use it
model = YOLO('../Model/yolov8l.pt')

# Global variables
mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
mqtt_port = 1883
mqtt_topic = "people_count"

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)
mqtt_client.loop_start()  # Start the MQTT loop

# MQTT publish function
def publish_count(camera_id, people_count):
    topic = f"{mqtt_topic}/{camera_id}"
    mqtt_client.publish(topic, str(people_count))

# Function to adjust ROI points based on provided coordinates
def set_roi_based_on_points(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points

# Function to capture and process frames for each camera in its own process
def capture_and_process_frames(camera_id, rtsp_url, coordinates):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for camera ID {camera_id}.")
        return

    roi_points = set_roi_based_on_points(coordinates["points"], coordinates)
    roi_mask = np.zeros((coordinates["display_height"], coordinates["display_width"]), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], (255, 255, 255))

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from RTSP stream for camera ID {camera_id}.")
            break

        # Resize the frame to the display size
        resized_frame = cv2.resize(frame, (coordinates["display_width"], coordinates["display_height"]))

        # Apply the ROI mask to the frame using bitwise AND
        imgRegion = cv2.bitwise_and(resized_frame, resized_frame, mask=roi_mask)

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
        if len(roi_points) > 1:
            cv2.polylines(resized_frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Show the annotated frame in a separate window for each camera
        cv2.imshow(f'People Count - Camera {camera_id}', resized_frame)

        # Publish the count to MQTT
        publish_count(camera_id, people_count)

        # Break the loop on 'q' key press or window close
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(f'People Count - Camera {camera_id}', cv2.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask route to start people counting
@app.route('/start_counting', methods=['POST'])
def start_counting():
    # Get JSON data from the request
    data = request.json

    processes = []

    # Loop through each camera configuration in the JSON input
    for camera_data in data:
        # Validate the input data
        required_keys = ['camera_id', 'rtsp_url', 'x', 'y', 'frame_width', 'frame_height', 'points']
        if not all(key in camera_data for key in required_keys):
            return jsonify({"error": f"Missing one or more required fields in JSON data for camera ID {camera_data.get('camera_id')}"}), 400

        # Extract the values
        camera_id = camera_data['camera_id']
        rtsp_url = camera_data['rtsp_url']
        x = camera_data['x']
        y = camera_data['y']
        display_width = camera_data['frame_width']
        display_height = camera_data['frame_height']
        points = camera_data['points']

        # Validate points
        if not isinstance(points, list) or len(points) < 3:
            return jsonify({"error": f"At least 3 points are required to define an ROI for camera ID {camera_id}"}), 400

        # Combine the extracted values into a coordinates dictionary
        coordinates = {
            "x": x,
            "y": y,
            "display_width": display_width,
            "display_height": display_height,
            "points": points
        }

        # Start the frame capture and processing in a separate process for each camera
        process = multiprocessing.Process(target=capture_and_process_frames, args=(camera_id, rtsp_url, coordinates))
        processes.append(process)
        process.start()

    return jsonify({"message": "People counting started successfully for all cameras"}), 200

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', port=5000)
