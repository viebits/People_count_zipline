import json
import multiprocessing
import cv2
import time
import math
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
from app.utils import capture_image, capture_video
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger
from app.exceptions import ziplineDetectionError

# Directories for saving images and videos
image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

tasks_processes = {}  # Dictionary to keep track of running processes


# Publish message to MQTT
def publish_message(motion_type, rtsp_link, site_id, camera_id, alarm_id, people_count, image):
    message = {
        "rtsp_link": rtsp_link,
        "siteId": site_id,
        "cameraId": camera_id,
        "alarmId": alarm_id,
        "type": motion_type,
        "people_count": people_count,
        "image": image
    }
    # mqtt_client.publish(mqtt_topic, json.dumps(message))
    print(f"Published message: {json.dumps(message)}")


# Initialize YOLO model and SORT tracker
model = YOLO("../Model/yolov8l.pt")
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)

# Global variables
detection_processes = {}


# Apply the given coordinates to ROI points
def use_coordinates(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    placed_points = []
    for point in points:
        placed_x = int(point[0] + x_offset)
        placed_y = int(point[1] + y_offset)
        placed_points.append((placed_x, placed_y))

    return placed_points


# Check if the centroid of a person is inside the ROI polygon
def is_inside_roi(cx, cy, roi_points):
    result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
    return result >= 0


# People counting logic based on ROI and zipline
def detect_people_count(rtsp_url, site_id, camera_id, alarm_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
    try:
        cap = cv2.VideoCapture(rtsp_url)
        count = 0  # Count variable for both up and down
        frame_skip = 3  # Process every Nth frame
        frame_count = 0

        previous_positions = {}
        debounce_time = 1.5  # seconds
        last_count_time = {}

        display_width, display_height = display_size
        buffer_vertical = 20  # Vertical buffer around the line
        buffer_horizontal = 20  # Horizontal buffer around the line

        arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
        arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))
        arrow_direction = arrow_coords['points'][1][1]  # Positive for down, negative for up

        window_name = f"Camera {camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        if not cap.isOpened():
            print(f"Error: Unable to open camera stream for camera {camera_id}")
            return

        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print(f"Error: Failed to read from camera {camera_id}")
                break

            frame = cv2.resize(frame, (display_width, display_height))
            cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)  # Draw the zipline
            cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)  # Draw the ROI
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)  # Draw the arrow
            cv2.imshow(window_name, frame)

            frame_count += 1
            if frame_count % frame_skip == 0:
                roi_mask = np.zeros_like(frame[:, :, 0])
                cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)
                roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

                results = model(roi_frame, stream=True)
                detections = np.empty((0, 5))
                tracking_started = False

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
                            detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
                            tracking_started = True

                if tracking_started:
                    tracked_people = tracker.update(detections)
                    m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
                    b = zipline_coords[0][1] - m * zipline_coords[0][0]

                    current_time = cv2.getTickCount() / cv2.getTickFrequency()

                    for person in tracked_people:
                        x1, y1, x2, y2, person_id = map(int, person)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box

                        if not is_inside_roi(cx, cy, roi_coords):
                            continue

                        zipline_y = m * cx + b
                        prev_cy = previous_positions.get(person_id, cy)

                        if (abs(cy - zipline_y) <= buffer_vertical) and (zipline_coords[0][0] - buffer_horizontal <= cx <= zipline_coords[1][0] + buffer_horizontal):
                            if person_id not in last_count_time or (current_time - last_count_time[person_id]) > debounce_time:
                                if arrow_direction < 0 and cy < zipline_y and prev_cy >= zipline_y:
                                    count += 1
                                    last_count_time[person_id] = current_time
                                    image_filename = capture_image(frame)
                                    publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, count, image_filename)
                                    cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 255, 0), 5)
                                elif arrow_direction > 0 and cy > zipline_y and prev_cy <= zipline_y:
                                    count += 1
                                    last_count_time[person_id] = current_time
                                    image_filename = capture_image(frame)
                                    publish_message("ZIP_LINE_CROSSING", rtsp_url, site_id, camera_id, alarm_id, count, image_filename)
                                    cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)

                        previous_positions[person_id] = cy
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                    cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        cap.release()
        cv2.destroyAllWindows()

        return count
    except Exception as e:
        logger.error(f"Error during Zipline crossing detection for camera {camera_id}: {str(e)}", exc_info=True)
        raise ziplineDetectionError(f"zipline detection failed for camera {camera_id}: {str(e)}")


# Function to start each camera in its own process
def zipline_start(task):
    """
        Start the zipline detection process in a separate thread for the given camera task.
        """
    try:
        camera_id = task["cameraId"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id] = stop_event

            # Start zipline detection in a new process
            process = multiprocessing.Process(
                target=detect_people_count,
                args=(task["rtsp_link"], camera_id, task["co_ordinates"], task["type"], stop_event)
            )
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started zipline detection for camera {camera_id}.")
        else:
            logger.warning(f"zipline detection already running for camera {camera_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True

def zipline_stop(camera_ids):
    """
    Stop zipline detection for the given camera IDs.
    """
    stopped_tasks = []
    not_found_tasks = []

    for camera_id in camera_ids:
        if camera_id in tasks_processes:
            try:
                tasks_processes[camera_id].terminate()  # Stop the process
                tasks_processes[camera_id].join()  # Wait for the process to stop
                del tasks_processes[camera_id]  # Remove from the dictionary
                stopped_tasks.append(camera_id)
                logger.info(f"Stopped zipline detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop zipline detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }





# @app.route('/zipline_start', methods=['POST'])
# def start_counting():
#     data = request.json
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera configurations"}), 400
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
#         arrow_coords = config['arrow']
#
#         detection_process, stop_event = start_camera_process(
#             rtsp_link, site_id, camera_id, alarm_id, roi_coords, line_coords, arrow_coords, (display_width, display_height))
#         detection_processes[camera_id] = {'process': detection_process, 'stop_event': stop_event}
#
#     return jsonify({"message": "People counting started successfully for all cameras"}), 200


# @app.route('/zipline_stop', methods=['POST'])
# def stop_counting():
#     data = request.json.get('camera_ids', [])
#     if not isinstance(data, list):
#         return jsonify({"error": "Input should be a list of camera IDs"}), 400
#
#     for camera_id in data:
#         if camera_id in detection_processes:
#             detection_processes[camera_id]['stop_event'].set()
#             detection_processes[camera_id]['process'].join()
#             del detection_processes[camera_id]
#
#     return jsonify({"message": "People counting stopped successfully for all cameras"}), 200
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
