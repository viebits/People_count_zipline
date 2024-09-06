# working properly for multiple cameras
from flask import Flask, request, jsonify
import numpy as np
import cv2
import threading
import math
from ultralytics import YOLO
from sort import Sort
from multiprocessing import Process, Event

app = Flask(__name__)

# Initialize tracker and model
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)
model = YOLO("../Model/yolov8l.pt")

# Dictionary to store ongoing processes and stop events
detection_processes = {}

# Track the previous positions of people to check crossing direction
previous_positions = {}


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


# People counting logic based on ROI and zipline
def detect_people_count(rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event):
    global previous_positions

    cap = cv2.VideoCapture(rtsp_url)
    people_count = 0
    frame_skip = 2  # To process every 2nd frame
    frame_count = 0

    display_width, display_height = display_size

    # Arrow start and end points based on arrow coordinates
    arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
    arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))

    # Create a window for each camera stream (no full-screen)
    window_name = f"Camera {camera_id}"
    cv2.namedWindow(window_name)

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break

        # Resize the frame to match the provided display size
        frame = cv2.resize(frame, (display_width, display_height))

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

                for person in tracked_people:
                    x1, y1, x2, y2, person_id = person
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calculate the centroid of the bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box

                    # Check if the centroid is inside the ROI
                    if not is_inside_roi(cx, cy, roi_coords):
                        continue  # Skip if the person is outside the ROI

                    # Check if the person crossed the zipline based on their current and previous positions
                    m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
                    b = zipline_coords[0][1] - m * zipline_coords[0][0]
                    zipline_y = m * cx + b

                    if person_id in previous_positions:
                        prev_cy = previous_positions[person_id]

                        # Check for "up" direction (from below the zipline to above it)
                        if arrow_coords['points'][1][1] < 0 and prev_cy > zipline_y and cy < zipline_y:
                            people_count += 1
                        # Check for "down" direction (from above the zipline to below it)
                        elif arrow_coords['points'][1][1] >= 0 and prev_cy < zipline_y and cy > zipline_y:
                            people_count += 1

                    # Update the person's previous position
                    previous_positions[person_id] = cy

                    # Draw bounding box and centroid on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                # Draw zipline and ROI on the frame using the coordinates directly from the JSON
                cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 3)
                cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)

                # Draw the arrow on the frame
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)

                # Show people count on the frame
                cv2.putText(frame, f"Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the frame in its own window for the camera
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()

    return people_count


# Function to start each camera in its own process
def start_camera_process(rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size):
    stop_event = Event()
    detection_process = Process(target=detect_people_count, args=(
    rtsp_url, camera_id, roi_coords, zipline_coords, arrow_coords, display_size, stop_event))
    detection_process.start()
    return detection_process, stop_event


# Flask route to start people counting detection for multiple cameras
@app.route('/start_counting', methods=['POST'])
def start_counting():
    data = request.json

    if not isinstance(data, list):
        return jsonify({"error": "Input should be a list of camera configurations"}), 400

    response = []

    for config in data:
        if 'rtsp_link' not in config:
            return jsonify({"error": "RTSP link is required for all configurations"}), 400

        rtsp_link = config['rtsp_link']
        camera_id = config['cameraId']
        display_width = config['display_width']
        display_height = config['display_height']
        line_coords = use_coordinates(config['line']['points'], config['line'])
        roi_coords = use_coordinates(config['roi']['points'], config['roi'])
        arrow_coords = config['arrow']  # Extract the arrow coordinates

        # Start a new process for each camera
        detection_process, stop_event = start_camera_process(rtsp_link, camera_id, roi_coords, line_coords,
                                                             arrow_coords, (display_width, display_height))
        detection_processes[camera_id] = {'process': detection_process, 'stop_event': stop_event}

        # Add status for each camera to the response
        response.append({
            "status": "Detection started",
            "camera_id": camera_id
        })

    return jsonify(response)


# Flask route to stop people counting detection for a specific camera
@app.route('/stop_counting/<int:camera_id>', methods=['POST'])
def stop_counting(camera_id):
    if camera_id in detection_processes:
        detection_processes[camera_id]['stop_event'].set()
        detection_processes[camera_id]['process'].join()
        del detection_processes[camera_id]
        return jsonify({"status": "Detection stopped", "camera_id": camera_id})
    else:
        return jsonify({"error": "Camera not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)



# # working properly with proper placement of roi and all for 1 camera
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
# # Function to directly use the provided points without scaling
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
#     # Create a full-screen window
#     window_name = f"Camera {camera_id}"
#     cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
#             # Run YOLO model on the frame
#             results = model(frame, stream=True)
#             detections = np.empty((0, 5))
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
#
#             # Track the people using SORT
#             tracked_people = tracker.update(detections)
#
#             for person in tracked_people:
#                 x1, y1, x2, y2, person_id = person
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box
#
#                 # Check if the person crossed the zipline based on their current and previous positions
#                 m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
#                 b = zipline_coords[0][1] - m * zipline_coords[0][0]
#                 zipline_y = m * cx + b
#
#                 if person_id in previous_positions:
#                     prev_cy = previous_positions[person_id]
#
#                     # Check for "up" direction (from below the zipline to above it)
#                     if arrow_coords['points'][1][1] < 0 and prev_cy > zipline_y and cy < zipline_y:
#                         people_count += 1
#                     # Check for "down" direction (from above the zipline to below it)
#                     elif arrow_coords['points'][1][1] >= 0 and prev_cy < zipline_y and cy > zipline_y:
#                         people_count += 1
#
#                 # Update the person's previous position
#                 previous_positions[person_id] = cy
#
#                 # Draw bounding box and zipline on the frame
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
#
#             # Draw zipline and ROI on the frame using the coordinates directly from the JSON
#             cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 3)
#             cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)
#
#             # Draw the arrow on the frame
#             cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)
#
#             # Show people count on the frame
#             cv2.putText(frame, f"Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # Show the frame in full screen
#             cv2.imshow(window_name, frame)
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
#     app.run(debug=True)



