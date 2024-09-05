# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# cap = cv2.VideoCapture("rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live")  # For Video
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
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
# # Variables to store drawn line and ROI
# line = None
# roi = None
# line_drawn = False
# roi_drawn = False
#
# # Callback functions for drawing
# def draw_line(event, x, y, flags, param):
#     global line, line_drawn
#     if event == cv2.EVENT_LBUTTONDOWN:
#         line = [(x, y)]
#     elif event == cv2.EVENT_LBUTTONUP:
#         line.append((x, y))
#         cv2.line(img_copy, line[0], line[1], (0, 0, 255), 3)
#         cv2.imshow('Line_Crossing', img_copy)
#         line_drawn = True  # Set the flag to True when line is drawn
#
# def select_roi(event, x, y, flags, param):
#     global roi, roi_drawn
#     if event == cv2.EVENT_LBUTTONDOWN:
#         roi = [(x, y)]
#     elif event == cv2.EVENT_LBUTTONUP:
#         roi.append((x, y))
#         cv2.rectangle(img_copy, roi[0], roi[1], (0, 255, 0), 2)
#         cv2.imshow('Line_Crossing', img_copy)
#         roi_drawn = True  # Set the flag to True when ROI is drawn
#
# # Let user draw line and select ROI
# ret, img = cap.read()
# img = cv2.resize(img, (1400, 700))
# img_copy = img.copy()
#
# cv2.imshow('Line_Crossing', img_copy)
# cv2.setMouseCallback('Line_Crossing', draw_line)
# print("Draw the line (click and drag). Press any key when done...")
# cv2.waitKey(0)
#
# cv2.setMouseCallback('Line_Crossing', select_roi)
# print("Select the ROI (click and drag). Press any key when done...")
# cv2.waitKey(0)
#
# cv2.destroyWindow('Line_Crossing')
#
# # Ensure both the line and ROI are drawn before starting detection
# if line_drawn and roi_drawn:
#     # Now proceed with the rest of your code using the selected line and ROI
#     frame_skip = 2  # Process every 2nd frame
#     frame_count = 0
#
#     width = 1400
#     height = 700
#
#     tracker_results = []
#
#     while True:
#         success, imgS = cap.read()
#         if not success:
#             break
#         img = cv2.resize(imgS, (width, height))
#
#         # Always draw the line and arrows
#         if line:
#             cv2.line(img, line[0], line[1], (0, 0, 255), 3)
#             arrow_length = 40  # The length of the arrows
#
#             # Down arrow (Red)
#             cv2.arrowedLine(img,
#                             (line[0][0] - 30, line[0][1] - arrow_length // 2),
#                             (line[0][0] - 30, line[0][1] + arrow_length // 2),
#                             (0, 0, 255), 3, tipLength=0.5)
#
#             # Up arrow (Green)
#             cv2.arrowedLine(img,
#                             (line[1][0] + 30, line[1][1] + arrow_length // 2),
#                             (line[1][0] + 30, line[1][1] - arrow_length // 2),
#                             (0, 255, 0), 3, tipLength=0.5)
#
#         if roi:
#             cv2.rectangle(img, roi[0], roi[1], (0, 255, 0), 2)
#
#         if frame_count % frame_skip == 0:
#             if roi:
#                 # Create a black mask and then draw a white rectangle over the ROI
#                 mask = np.zeros_like(img)
#                 x1, y1 = roi[0]
#                 x2, y2 = roi[1]
#                 cv2.rectangle(mask, roi[0], roi[1], (255, 255, 255), -1)
#
#                 # Apply the mask to the image using bitwise AND
#                 masked_img = cv2.bitwise_and(img, mask)
#
#                 # Perform detection on the masked image
#                 results = model(masked_img, stream=True)
#
#                 detections = np.empty((0, 5))
#
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         # Bounding Box
#                         bx1, by1, bx2, by2 = box.xyxy[0]
#                         bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
#                         bw, bh = bx2 - bx1, by2 - by1
#
#                         # Map back to original coordinates
#                         x1_full, y1_full, x2_full, y2_full = bx1, by1, bx2, by2
#
#                         # Confidence
#                         conf = math.ceil((box.conf[0] * 100)) / 100
#                         # Class Name
#                         cls = int(box.cls[0])
#                         currentClass = classNames[cls]
#
#                         if currentClass == "person" and conf > 0.3:
#                             currentArray = np.array([x1_full, y1_full, x2_full, y2_full, conf])
#                             detections = np.vstack((detections, currentArray))
#             else:
#                 # Perform detection on the entire image if no ROI is defined
#                 results = model(img, stream=True)
#
#                 detections = np.empty((0, 5))
#
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         # Bounding Box
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                         w, h = x2 - x1, y2 - y1
#
#                         # Confidence
#                         conf = math.ceil((box.conf[0] * 100)) / 100
#                         # Class Name
#                         cls = int(box.cls[0])
#                         currentClass = classNames[cls]
#
#                         if currentClass == "person" and conf > 0.3:
#                             currentArray = np.array([x1, y1, x2, y2, conf])
#                             detections = np.vstack((detections, currentArray))
#
#             tracker_results = tracker.update(detections)
#
#             # Calculate the slope (m) and intercept (b) of the line
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
#
#                 # Calculate the y-value on the line at the object's x-coordinate (cx)
#                 line_y_at_cx = m * cx + b
#
#                 # Restrict detection to objects within the buffer zone of the line
#                 if (abs(cy - line_y_at_cx) <= buffer_vertical) and (line[0][0] - buffer_horizontal <= cx <= line[1][0] + buffer_horizontal):
#                     if id not in last_count_time or (current_time - last_count_time[id]) > debounce_time:
#                         # Check for upward crossing
#                         if cy < line_y_at_cx and previous_y >= line_y_at_cx:
#                             if not crossedPersons[id]['crossed_up']:
#                                 totalCountUp.append(id)
#                                 crossedPersons[id]['crossed_up'] = True
#                                 crossedPersons[id]['crossed_down'] = False
#                                 cv2.line(img, line[0], line[1], (0, 255, 0), 5)  # Change line color to green for up
#                                 last_count_time[id] = current_time
#
#                         # Check for downward crossing
#                         elif cy > line_y_at_cx and previous_y <= line_y_at_cx:
#                             if not crossedPersons[id]['crossed_down']:
#                                 totalCountDown.append(id)
#                                 crossedPersons[id]['crossed_down'] = True
#                                 crossedPersons[id]['crossed_up'] = False
#                                 cv2.line(img, line[0], line[1], (0, 0, 255), 5)  # Keep line color red for down
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
#         # Show the image
#         cv2.imshow("Line_Crossing", img)
#
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Line_Crossing', cv2.WND_PROP_VISIBLE) < 1):
#             break
#
#         frame_count += 1
#
#     cap.release()
#     cv2.destroyAllWindows()
# else:
#     print("Both line and ROI must be defined to start processing.")

#



# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# cap = cv2.VideoCapture("rtsp://admin:Admin@123@36.255.211.155:554/media/video2")  # For Video
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
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
# line = [100, 350, 950, 500]  # x1, y1, x2, y2 for the line
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
# # mask = cv2.imread("mask.png")
#
# frame_skip = 1  # Process every 2nd frame
# frame_count = 0
#
# width = 1400
# height = 700
#
# # Store tracker results to use for visualization on every frame
# tracker_results = []
#
# while True:
#     success, imgS = cap.read()
#     if not success:
#         break
#     img = cv2.resize(imgS, (width, height))
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
#         # imgRegion = cv2.bitwise_and(img, mask)
#         results = model(img, stream=True)
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
#         tracker_results = tracker.update(detections)
#
#         # Calculate the slope (m) and intercept (b) of the line
#         m = (line[3] - line[1]) / (line[2] - line[0])  # slope of the line
#         b = line[1] - m * line[0]  # y-intercept of the line
#
#         current_time = cv2.getTickCount() / cv2.getTickFrequency()
#
#         for result in tracker_results:
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
#     else:
#         # Draw tracker results on non-processing frames
#         for result in tracker_results:
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


# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# cap = cv2.VideoCapture("rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live")  # For Video
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
# # Check if the mask is loaded properly
# if mask is None:
#     print("Error: Mask image could not be loaded. Please check the file path.")
#     exit()
#
# while True:
#     success, img = cap.read()
#
#     if not success:
#         print("Error: Could not read frame from the video stream.")
#         break
#
#     # Ensure that the mask is resized to match the dimensions of the image
#     if mask.shape[:2] != img.shape[:2]:
#         mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#
#     imgRegion = cv2.bitwise_and(img, mask)
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
#         if (abs(cy - line_y_at_cx) <= buffer_vertical) and (
#                 line[0] - buffer_horizontal <= cx <= line[2] + buffer_horizontal):
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
#                         cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255),
#                                  5)  # Keep line color red for down
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

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Load video stream
cap = cv2.VideoCapture("rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live")

# Load the YOLO model
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

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)

# Line coordinates (draw by default)
line = [270, 180, 450, 160]

totalCountUp = []
totalCountDown = []

# Track which direction is being counted
selected_direction = None

# Mouse event callback for drawing an arrow
drawing = False
start_point = None
end_point = None

def draw_arrow(event, x, y, flags, param):
    global drawing, start_point, end_point, selected_direction

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        # Determine the arrow direction (up or down)
        if start_point and end_point:
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]

            if dy < 0:
                selected_direction = 'up'
            elif dy > 0:
                selected_direction = 'down'

            start_point = None
            end_point = None

# Mask image for restricting region
mask = cv2.imread("mask.png")
if mask is None:
    print("Error: Mask image could not be loaded.")
    exit()

# Create a window and bind the mouse callback
cv2.namedWindow("Line_Crossing")
cv2.setMouseCallback("Line_Crossing", draw_arrow)

while True:
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame from the video stream.")
        break

    # Resize mask to match image dimensions
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

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

    # Draw the line
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check crossing logic based on selected direction
        if selected_direction == 'up':
            # Count only upward crossings
            if cy < line[1]:
                totalCountUp.append(id)
        elif selected_direction == 'down':
            # Count only downward crossings
            if cy > line[1]:
                totalCountDown.append(id)

    # Display counts based on selected direction
    if selected_direction == 'up':
        cv2.putText(img, f'UP: {len(totalCountUp)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    elif selected_direction == 'down':
        cv2.putText(img, f'DOWN: {len(totalCountDown)}', (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    # Show the user-drawn arrow
    if start_point and end_point:
        cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 3, tipLength=0.5)

    # Show the image
    cv2.imshow("Line_Crossing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


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
