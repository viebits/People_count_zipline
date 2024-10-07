import json
import multiprocessing
import cv2
import time
import os
import numpy as np
from app.utils import capture_image, capture_video
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger
from app.exceptions import MotionDetectionError

tasks_processes = {}  # Dictionary to keep track of running processes


def set_roi_based_on_points(points, coordinates):
    """
    Scale and set ROI based on given points and coordinates.
    """
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))
    return scaled_points


def detect_motion(rtsp_url, camera_id, coordinates, motion_type, siteid, stop_event):
    """
    Motion detection loop that captures video frames, detects motion in the ROI,
    and captures images and videos when motion is detected.
    """
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer size

        # Parameters for motion detection
        threshold_value = 16
        min_area_full_frame = 1200

        # Initialize first frame
        ret, frame = cap.read()
        if not ret:
            raise MotionDetectionError(f"Failed to read the stream from camera {camera_id}.")

        # Get display width and height from coordinates
        display_width = coordinates["display_width"]
        display_height = coordinates["display_height"]

        # Resize the frame to match display size
        frame = cv2.resize(frame, (display_width, display_height))

        if coordinates['points']:
            roi_points = set_roi_based_on_points(coordinates["points"], coordinates)

            # Convert roi_points to a numpy array
            roi_points = np.array(roi_points, dtype=np.int32)

            # Create ROI mask
            roi_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], (255, 255, 255))

            # Calculate the area of the ROI
            roi_area = cv2.countNonZero(roi_mask)
            full_frame_area = display_width * display_height
            min_area = (min_area_full_frame / full_frame_area) * roi_area
        else:
            min_area = min_area_full_frame
            roi_mask = None

        # Initialize previous frame for comparison
        prev_frame_gray = None
        last_detection_time = 0
        window_name = f"Motion Detection - Camera {camera_id}"

        logger.info(f"Motion detection started for camera {camera_id}.")

        while not stop_event.is_set():
           # Resize frame to match display size
            frame = cv2.resize(frame, (display_width, display_height))
            display_frame = frame.copy()

            if roi_mask is not None:
                # Apply ROI mask to the frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                # Draw all ROIs on the display frame
                cv2.polylines(display_frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
            else:
                masked_frame = frame

            # Convert frame to grayscale and apply Gaussian blur
            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if prev_frame_gray is None:
                prev_frame_gray = gray_frame
                continue

            # Calculate frame difference and threshold
            frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
            _, thresh_frame = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            # Find contours in the threshold frame
            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    motion_detected = True

            # Motion detection handling
            current_time = time.time()
            if motion_detected and (current_time - last_detection_time > 60):
                logger.info(f"Motion detected for camera {camera_id}. Capturing image and video.")

                # Capture image and video
                frame_copy = frame.copy()
                image_filename = capture_image(frame_copy)
                video_filename = "testing" # capture_video(rtsp_url)

                try:
                    message = {
                        "rtsp_link": rtsp_url,
                        "cameraId": camera_id,
                        "type": motion_type,
                        "siteId": siteid,
                        "image": image_filename,
                        "video": video_filename
                    }
                    pub("motion/detection", message)
                except Exception as e:
                    logger.error(f"Error publishing MQTT message: {e}", exc_info=True)
                    raise

                last_detection_time = current_time

            # Display frame
            cv2.imshow(window_name, display_frame)

            # Update previous frame
            prev_frame_gray = gray_frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyWindow(window_name)

    except Exception as e:
        logger.error(f"Error during motion detection for camera {camera_id}: {str(e)}", exc_info=True)
        raise MotionDetectionError(f"Motion detection failed for camera {camera_id}: {str(e)}")


def motion_start(task):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        camera_id = task["cameraId"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id] = stop_event

            # Start motion detection in a new process
            process = multiprocessing.Process(
                target=detect_motion,
                args=(task["rtsp_link"], camera_id, task["co_ordinates"], task["type"], task["siteId"], stop_event)
            )
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started motion detection for camera {camera_id}.")
        else:
            logger.warning(f"Motion detection already running for camera {camera_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True

def motion_stop(camera_ids):
    """
    Stopmotion detection for the given camera IDs.
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
                logger.info(f"Stopped motion detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop motion detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }
