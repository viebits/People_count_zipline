# api with start and stop
import cv2
import multiprocessing
import numpy as np
import json
from ultralytics import YOLO
from flask import Flask, jsonify, request
import paho.mqtt.client as mqtt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the YOLOv8 model globally so each process can use it
model = YOLO('../Model/yolov8l.pt')

# Global variables
mqtt_broker = "192.168.1.120"  # Replace with your MQTT broker address
mqtt_port = 1883
mqtt_topic = "peoplecount/detected"

# Dictionary to store the active processes
active_processes = {}

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)
mqtt_client.loop_forever()  # Start the MQTT loop

# MQTT publish function
def publish_count(camera_id, people_count):
    topic = f"{mqtt_topic}"
    mqtt_client.publish(topic, str(people_count))

def publish_message(motion_type, rtsp_link, site_id, camera_id, alarm, people_count):
    message = {
        "rtsp_link": rtsp_link,
        "siteId": site_id,
        "cameraId": camera_id,
        "type": motion_type,
        "people_count": people_count,
    }
    mqtt_client.publish(mqtt_topic, json.dumps(message))
    print(f"Published message: {json.dumps(message)}")

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
def capture_and_process_frames(cameraId, rtsp_link, site_id, alarmId, coordinates):
    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for camera ID {cameraId}.")
        return

    roi_points = set_roi_based_on_points(coordinates["points"], coordinates)
    roi_mask = np.zeros((coordinates["display_height"], coordinates["display_width"]), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], (255, 255, 255))

    previous_people_count = None  # To track the previous count

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from RTSP stream for camera ID {cameraId}.")
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
        cv2.imshow(f'People Count - Camera {cameraId}', resized_frame)

        # Publish the count to MQTT only if it has changed
        if previous_people_count is None or previous_people_count != people_count:
            publish_message("PEOPLE_COUNT", rtsp_link, site_id, cameraId, alarmId, people_count)
            previous_people_count = people_count  # Update the previous count

        # Break the loop on 'q' key press or window close
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(f'People Count - Camera {cameraId}', cv2.WND_PROP_VISIBLE) < 1):
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
        print(camera_data, '-----------------camera_data')
        # Extract the values
        cameraId = camera_data['cameraId']
        alarmId = camera_data['alarmId']
        rtsp_link = camera_data['rtsp_link']
        x = camera_data['x']
        y = camera_data['y']
        display_width = camera_data['display_width']
        display_height = camera_data['display_height']
        points = camera_data['points']
        site_id = camera_data['siteId']

        # Validate points
        if not isinstance(points, list) or len(points) < 3:
            return jsonify({"error": f"At least 3 points are required to define an ROI for camera ID {cameraId}"}), 400

        # Combine the extracted values into a coordinates dictionary
        coordinates = {
            "x": x,
            "y": y,
            "display_width": display_width,
            "display_height": display_height,
            "points": points
        }

        # Start the frame capture and processing in a separate process for each camera
        process = multiprocessing.Process(target=capture_and_process_frames, args=(cameraId, rtsp_link, site_id, alarmId, coordinates))
        processes.append(process)
        process.start()

        # Store the process in the global dictionary
        active_processes[cameraId] = process

    return jsonify({"message": "People counting started successfully for all cameras"}), 200

# Flask route to stop people counting for specific cameras
@app.route('/stop_counting', methods=['POST'])
def stop_counting():
    data = request.json
    camera_ids = data.get('camera_ids', [])

    for camera_id in camera_ids:
        process = active_processes.get(camera_id)
        if process and process.is_alive():
            process.terminate()  # Terminate the process
            process.join()  # Ensure process has finished
            del active_processes[camera_id]  # Remove the process from the dictionary
            print(f"Stopped and terminated process for camera ID {camera_id}.")
        else:
            print(f"No active process found for camera ID {camera_id}.")

    return jsonify({"message": f"Stopped processes for camera IDs: {camera_ids}"}), 200

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', port=5000)
