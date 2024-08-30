import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("rtsp://admin:InfosolDev@123@103.162.197.92:602/media/video2")  # For Video
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

model = YOLO("../Model/yolov8l.pt")

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

# Initialize tracking with adjusted IOU threshold
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)  # Increased IOU threshold to reduce overlaps

# Line coordinates (single line)
line = [270, 180, 440, 160]  # x1, y1, x2, y2 for the line

totalCountUp = []
totalCountDown = []

# Track persons that have crossed the line
crossedPersons = {}

# Define a buffer zone around the line
buffer_vertical = 50  # Vertical buffer around the line
buffer_horizontal = 50  # Horizontal buffer around the line

# Debounce mechanism: track the last time a person was counted to prevent rapid re-counting
debounce_time = 1.5  # seconds
last_count_time = {}

mask = cv2.imread("mask.png")

frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Store tracker results to use for visualization on every frame
tracker_results = []

while True:
    success, img = cap.read()
    if not success:
        break

    # Always draw the line and arrows
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
    arrow_length = 40  # The length of the arrows

    # Down arrow (Red)
    cv2.arrowedLine(img,
                    (line[0] - 30, line[1] - arrow_length // 2),
                    (line[0] - 30, line[1] + arrow_length // 2),
                    (0, 0, 255), 3, tipLength=0.5)

    # Up arrow (Green)
    cv2.arrowedLine(img,
                    (line[2] + 30, line[3] + arrow_length // 2),
                    (line[2] + 30, line[3] - arrow_length // 2),
                    (0, 255, 0), 3, tipLength=0.5)

    if frame_count % frame_skip == 0:
        imgRegion = cv2.bitwise_and(img, mask)
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        tracker_results = tracker.update(detections)

        # Calculate the slope (m) and intercept (b) of the line
        m = (line[3] - line[1]) / (line[2] - line[0])  # slope of the line
        b = line[1] - m * line[0]  # y-intercept of the line

        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        for result in tracker_results:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if id not in crossedPersons:
                crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}

            previous_y = crossedPersons[id]['previous_y']

            # Calculate the y-value on the line at the object's x-coordinate (cx)
            line_y_at_cx = m * cx + b

            # Restrict detection to objects within the buffer zone of the line
            if (abs(cy - line_y_at_cx) <= buffer_vertical) and (line[0] - buffer_horizontal <= cx <= line[2] + buffer_horizontal):
                if id not in last_count_time or (current_time - last_count_time[id]) > debounce_time:
                    # Check for upward crossing
                    if cy < line_y_at_cx and previous_y >= line_y_at_cx:
                        if not crossedPersons[id]['crossed_up']:
                            totalCountUp.append(id)
                            crossedPersons[id]['crossed_up'] = True
                            crossedPersons[id]['crossed_down'] = False
                            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0),
                                     5)  # Change line color to green for up
                            last_count_time[id] = current_time

                    # Check for downward crossing
                    elif cy > line_y_at_cx and previous_y <= line_y_at_cx:
                        if not crossedPersons[id]['crossed_down']:
                            totalCountDown.append(id)
                            crossedPersons[id]['crossed_down'] = True
                            crossedPersons[id]['crossed_up'] = False
                            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)  # Keep line color red for down
                            last_count_time[id] = current_time

            crossedPersons[id]['previous_y'] = cy

    else:
        # Draw tracker results on non-processing frames
        for result in tracker_results:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Display counts with labels
    cv2.putText(img, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.putText(img, f'DOWN: {len(totalCountDown)}', (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    # Show the image
    cv2.imshow("Line_Crossing", img)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Line_Crossing', cv2.WND_PROP_VISIBLE) < 1):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()




# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# cap = cv2.VideoCapture("rtsp://admin:InfosolDev@123@103.162.197.92:602/media/video2")  # For Video
#
# model = YOLO("../Model/yolov8l.pt")
#
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
# # Initialize tracking with adjusted IOU threshold
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)  # Increased IOU threshold to reduce overlaps
#
# # Line coordinates (single line)
# line = [270, 180, 450, 160]  # x1, y1, x2, y2 for the line
#
# totalCountUp = []
# totalCountDown = []
#
# # Track persons that have crossed the line
# crossedPersons = {}
#
# # Define a buffer zone around the line
# buffer_vertical = 50  # Vertical buffer around the line
# buffer_horizontal = 50  # Horizontal buffer around the line
#
# # Debounce mechanism: track the last time a person was counted to prevent rapid re-counting
# debounce_time = 1.5  # seconds
# last_count_time = {}
#
# mask = cv2.imread("mask.png")
#
# while True:
#     success, img = cap.read()
#     imgRegion = cv2.bitwise_and(img,mask)
#     results = model(imgRegion, stream=True)
#
#     detections = np.empty((0, 5))
#
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#             currentClass = classNames[cls]
#
#             if currentClass == "person" and conf > 0.3:
#                 currentArray = np.array([x1, y1, x2, y2, conf])
#                 detections = np.vstack((detections, currentArray))
#
#     resultsTracker = tracker.update(detections)
#
#     # Draw the line
#     cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
#
#     arrow_length = 40  # The length of the arrows
#
#     # Down arrow (Red)
#     cv2.arrowedLine(img,
#                     (line[0] - 30, line[1] - arrow_length // 2),
#                     (line[0] - 30, line[1] + arrow_length // 2),
#                     (0, 0, 255), 3, tipLength=0.5)
#
#     # Up arrow (Green)
#     cv2.arrowedLine(img,
#                     (line[2] + 30, line[3] + arrow_length // 2),
#                     (line[2] + 30, line[3] - arrow_length // 2),
#                     (0, 255, 0), 3, tipLength=0.5)
#
#     # Calculate the slope (m) and intercept (b) of the line
#     m = (line[3] - line[1]) / (line[2] - line[0])  # slope of the line
#     b = line[1] - m * line[0]  # y-intercept of the line
#
#     current_time = cv2.getTickCount() / cv2.getTickFrequency()
#
#     for result in resultsTracker:
#         x1, y1, x2, y2, id = result
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = x1 + w // 2, y1 + h // 2
#
#         cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#         cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
#                            scale=2, thickness=3, offset=10)
#
#         cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#         if id not in crossedPersons:
#             crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}
#
#         previous_y = crossedPersons[id]['previous_y']
#
#         # Calculate the y-value on the line at the object's x-coordinate (cx)
#         line_y_at_cx = m * cx + b
#
#         # Restrict detection to objects within the buffer zone of the line
#         if (abs(cy - line_y_at_cx) <= buffer_vertical) and (line[0] - buffer_horizontal <= cx <= line[2] + buffer_horizontal):
#             if id not in last_count_time or (current_time - last_count_time[id]) > debounce_time:
#                 # Check for upward crossing
#                 if cy < line_y_at_cx and previous_y >= line_y_at_cx:
#                     if not crossedPersons[id]['crossed_up']:
#                         totalCountUp.append(id)
#                         crossedPersons[id]['crossed_up'] = True
#                         crossedPersons[id]['crossed_down'] = False
#                         cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0),
#                                  5)  # Change line color to green for up
#                         last_count_time[id] = current_time
#
#                 # Check for downward crossing
#                 elif cy > line_y_at_cx and previous_y <= line_y_at_cx:
#                     if not crossedPersons[id]['crossed_down']:
#                         totalCountDown.append(id)
#                         crossedPersons[id]['crossed_down'] = True
#                         crossedPersons[id]['crossed_up'] = False
#                         cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)  # Keep line color red for down
#                         last_count_time[id] = current_time
#
#         crossedPersons[id]['previous_y'] = cy
#
#     # Display counts with labels
#     cv2.putText(img, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
#     cv2.putText(img, f'DOWN: {len(totalCountDown)}', (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#
#     # Show the image
#     cv2.imshow("Line_Crossing", img)
#
#     if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Line_Crossing', cv2.WND_PROP_VISIBLE) < 1):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


#Tracker for skipping frames
# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# cap = cv2.VideoCapture("rtsp://admin:InfosolDev@123@103.162.197.92:602/media/video2")  # For Video
#
# model = YOLO("../Model/yolov8l.pt")
#
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
# # Initialize tracking with adjusted IOU threshold
# tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)  # Increased IOU threshold to reduce overlaps
#
# # Line coordinates (single line)
# line = [270, 180, 450, 160]  # x1, y1, x2, y2 for the line
#
# totalCountUp = []
# totalCountDown = []
#
# # Track persons that have crossed the line
# crossedPersons = {}
#
# # Define a buffer zone around the line
# buffer_vertical = 50  # Vertical buffer around the line
# buffer_horizontal = 50  # Horizontal buffer around the line
#
# # Debounce mechanism: track the last time a person was counted to prevent rapid re-counting
# debounce_time = 1.5  # seconds
# last_count_time = {}
#
# mask = cv2.imread("mask.png")
#
# frame_skip = 2  # Process every 2nd frame
# frame_count = 0
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     # Always draw the line and arrows
#     cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
#     arrow_length = 40  # The length of the arrows
#
#     # Down arrow (Red)
#     cv2.arrowedLine(img,
#                     (line[0] - 30, line[1] - arrow_length // 2),
#                     (line[0] - 30, line[1] + arrow_length // 2),
#                     (0, 0, 255), 3, tipLength=0.5)
#
#     # Up arrow (Green)
#     cv2.arrowedLine(img,
#                     (line[2] + 30, line[3] + arrow_length // 2),
#                     (line[2] + 30, line[3] - arrow_length // 2),
#                     (0, 255, 0), 3, tipLength=0.5)
#
#     if frame_count % frame_skip == 0:
#         imgRegion = cv2.bitwise_and(img, mask)
#         results = model(imgRegion, stream=True)
#
#         detections = np.empty((0, 5))
#
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#
#                 # Confidence
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 # Class Name
#                 cls = int(box.cls[0])
#                 currentClass = classNames[cls]
#
#                 if currentClass == "person" and conf > 0.3:
#                     currentArray = np.array([x1, y1, x2, y2, conf])
#                     detections = np.vstack((detections, currentArray))
#
#         resultsTracker = tracker.update(detections)
#
#         # Calculate the slope (m) and intercept (b) of the line
#         m = (line[3] - line[1]) / (line[2] - line[0])  # slope of the line
#         b = line[1] - m * line[0]  # y-intercept of the line
#
#         current_time = cv2.getTickCount() / cv2.getTickFrequency()
#
#         for result in resultsTracker:
#             x1, y1, x2, y2, id = result
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             cx, cy = x1 + w // 2, y1 + h // 2
#
#             cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#             cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
#                                scale=2, thickness=3, offset=10)
#
#             cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#             if id not in crossedPersons:
#                 crossedPersons[id] = {'crossed_up': False, 'crossed_down': False, 'previous_y': cy}
#
#             previous_y = crossedPersons[id]['previous_y']
#
#             # Calculate the y-value on the line at the object's x-coordinate (cx)
#             line_y_at_cx = m * cx + b
#
#             # Restrict detection to objects within the buffer zone of the line
#             if (abs(cy - line_y_at_cx) <= buffer_vertical) and (line[0] - buffer_horizontal <= cx <= line[2] + buffer_horizontal):
#                 if id not in last_count_time or (current_time - last_count_time[id]) > debounce_time:
#                     # Check for upward crossing
#                     if cy < line_y_at_cx and previous_y >= line_y_at_cx:
#                         if not crossedPersons[id]['crossed_up']:
#                             totalCountUp.append(id)
#                             crossedPersons[id]['crossed_up'] = True
#                             crossedPersons[id]['crossed_down'] = False
#                             cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0),
#                                      5)  # Change line color to green for up
#                             last_count_time[id] = current_time
#
#                     # Check for downward crossing
#                     elif cy > line_y_at_cx and previous_y <= line_y_at_cx:
#                         if not crossedPersons[id]['crossed_down']:
#                             totalCountDown.append(id)
#                             crossedPersons[id]['crossed_down'] = True
#                             crossedPersons[id]['crossed_up'] = False
#                             cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)  # Keep line color red for down
#                             last_count_time[id] = current_time
#
#             crossedPersons[id]['previous_y'] = cy
#
#     # Display counts with labels
#     cv2.putText(img, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
#     cv2.putText(img, f'DOWN: {len(totalCountDown)}', (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#
#     # Show the image
#     cv2.imshow("Line_Crossing", img)
#
#     if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Line_Crossing', cv2.WND_PROP_VISIBLE) < 1):
#         break
#
#     frame_count += 1
#
# cap.release()
# cv2.destroyAllWindows()
