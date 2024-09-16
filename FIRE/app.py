import cv2
from skimage.data import camera
from ultralytics import YOLO
import os
import multiprocessing
import time
from app.utils import capture_image, capture_video  # Assuming capture_image and capture_video are defined in utils
from app.mqtt_handler import publish_message_mqtt as pub  # Assuming you have an MQTT handler setup
from app.config import logger  # Assuming you have a logger setup in config
from app.exceptions import Firedetectionerror
# Directory to save images or videos if needed
image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

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

# RTSP Stream URL (replace with your actual RTSP URL)
rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'
tasks_processes = {}  # Dictionary to keep track of running processes


def detect_fire(rtsp_url, camera_id, coordinates, motion_type, stop_event):
    """
    Fire detection function that captures video frames, performs inference,
    and publishes a message if fire is detected.
    """
    try:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise Exception(f"Error: Unable to open RTSP stream at {rtsp_url}")

        # Load the YOLO model for fire detection
        model_path = 'best.pt'  # Replace with the path to your YOLOv8 best.pt model
        model = YOLO(model_path)

        # Frame counter for skipping logic
        frame_skip_interval = 3
        frame_counter = 0
        last_detection_time=0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to grab frame from RTSP stream {rtsp_url}")
                break

            # Skip frames based on the frame_skip_interval
            frame_counter += 1
            if frame_counter % frame_skip_interval != 0:
                continue

            # Get display width and height from coordinates
            display_width = coordinates["display_width"]
            display_height = coordinates["display_height"]

            # Resize the frame to match display size
            frame = cv2.resize(frame, (display_width, display_height))

            # Perform inference on every Nth frame
            results = model(frame)

            # Annotate detected objects with class names and bounding boxes
            fire_detected = False
            for result in results.boxes:
                class_id = int(result.cls[0])
                confidence = result.conf[0]
                x1, y1, x2, y2 = map(int, result.xyxy[0])

                # Get the class name from the model's class names
                class_name = model.names[class_id]
                logger.info(f"Detected class: {class_name}, Confidence: {confidence:.2f}")

                # Only trigger for "fire"
                if class_name.lower() == "fire":
                    fire_detected = True
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{class_name} ({confidence:.2f})',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # If fire is detected, publish a message
            current_time=time.time()
            if fire_detected and (current_time - last_detection_time > 60):
                logger.info(f"Fire detected for camera {camera_id}. Capturing image and video.")

                frame_copy=frame.copy()
                image_path = capture_image(frame_copy)  # Save an image when fire is detected
                video_path = capture_video(rtsp_url)  # Capture video from the stream


                try:
                    # Publish to MQTT
                    pub_message = {
                    "rtsp_link": rtsp_url,
                    "type": "fire",
                    "image": image_path,
                    "video": video_path
                    }
                    pub("fire/detection", pub_message)
                    logger.info(f"Published fire detection message: {pub_message}")
                except Exception as e:
                    logger.error(f"Error publishing MQTT message: {e}", exc_info=True)
                    raise

                last_detection_time=current_time

            # Display the frame
            cv2.imshow('Fire Detection (RTSP)', frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during Fire detection for camera {camera_id}: {str(e)}", exc_info=True)
        raise Firedetectionerror(f"Fire detection failed for camera {camera_id}: {str(e)}")

def fire_detection_start(task):
    """
    Start the fire detection process.
    """
    try:
        camera_id=task["camera_id"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id]=stop_event
            # start fire detection
            process = multiprocessing.Process(
                target=detect_fire,
                args=(task["rtsp_link"], camera_id, task["co_ordinates"], task["type"], stop_event)
            )
            tasks_processes["fire_detection"] = process
            process.start()
            logger.info(f"Started Fire detection for camera {camera_id}.")
        else:
            logger.warning(f"Fire detection already running for camera {camera_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True



def fire_detection_stop(camera_ids):
    """
    Stop Fire detection for the given camera IDs.
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
                logger.info(f"Stopped Fire detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop Fire detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }


# if __name__ == '__main__':
#     # Start fire detection process
#     fire_detection_start()
#
#     # You can add logic to stop the process after some time or based on user input
#     # Example to stop after 1 minute:
#     time.sleep(60)
#     fire_detection_stop()
