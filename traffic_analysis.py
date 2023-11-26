#!/usr/bin/env python
# coding: utf-8

# In[2]:


# traffic_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# def analyze_traffic(image_path):
#     weights_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.weights"
#     config_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.cfg"
#     names_path = "C:\\Users\\Shalv Srivastava\\yolov3\\coco.names"

#     net = cv2.dnn.readNet(weights_path, config_path)

#     with open(names_path, 'r') as f:
#         classes = f.read().strip().split('\n')

#     indian_vehicle_classes = ['car', 'bike', 'truck', 'bus', 'motorbike', 'bicycle', 'rickshaw', 'cycle']

#     classes.extend(indian_vehicle_classes)

#     confidence_thresholds = {
#         'car': 0.6,
#         'bike': 0.2,
#         'truck': 0.9,
#         'bus': 0.9,
#         'motorbike': 0.4,
#         'bicycle': 0.4,
#         'rickshaw': 0.3,
#         'cycle': 0.3
#     }

#     def detect_objects(image):
#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#         net.setInput(blob)
#         output_layers_names = net.getUnconnectedOutLayersNames()
#         outputs = net.forward(output_layers_names)

#         vehicle_count = {class_name: 0 for class_name in indian_vehicle_classes}
#         vehicle_classes = []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = int(np.argmax(scores))
#                 confidence = scores[class_id]

#                 class_name = classes[class_id]

#                 if class_name in indian_vehicle_classes and confidence > confidence_thresholds[class_name]:
#                     if class_name in indian_vehicle_classes and vehicle_count[class_name] < 30:
#                         vehicle_count[class_name] += 1

#                     vehicle_classes.append(class_name)
#                     center_x = int(detection[0] * image.shape[1])
#                     center_y = int(detection[1] * image.shape[0])
#                     w = int(detection[2] * image.shape[1])
#                     h = int(detection[3] * image.shape[0])

#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Convert the image to base64 with bounding boxes
#         _, buffer = cv2.imencode('.png', image)
#         image_base64 = base64.b64encode(buffer).decode('utf-8')

#         return image_base64, vehicle_count, vehicle_classes

#     image = cv2.imread(image_path)
#     result_image, vehicle_count, vehicle_classes = detect_objects(image)

#     return result_image, vehicle_count, vehicle_classes


    
#     #cv2.imshow("Image", image)
#     #cv2.waitKey(0)
#     #cv2.destroyAllWindows()

#     cv2.imshow("Result", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     def detect_lanes(image, sf=0.01):
#         plt.figure(figsize=(image.shape[1]/80, image.shape[0]/80), dpi=80)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
#         edges = cv2.Canny(image_blur, 50, 150)
#         height, width = image.shape[:2]
#         roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
#         mask = np.zeros_like(edges)
#         cv2.fillPoly(mask, [np.array(roi_vertices)], 255)
#         masked_edges = cv2.bitwise_and(edges, mask)
#         lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

#         if lines is not None:
#             lanes_image = np.zeros_like(image)
#             lane_dimensions = []
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(lanes_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 lane_dimensions.append(((x1, y1), (x2, y2)))

#             lanes_area_pixels = sum((abs(x2 - x1) * abs(y2 - y1)) for (x1, y1), (x2, y2) in lane_dimensions)
#             total_area_input_image = height * width
#             real_area = (lanes_area_pixels * sf**2) / (lanes_area_pixels / total_area_input_image)

#             width = (lanes_area_pixels / 2.5) ** 0.5

#             print(f"Total Lane Area:")
#             print(f"In Square Pixels: {lanes_area_pixels} pixels^2")
#             print(f"Real Area: {real_area} units^2")
#             print(f"Calculated Width: {width} units")

#             return lanes_image, lanes_area_pixels

#         else:
#             lanes_image = image.copy()
#             return lanes_image, 0

#     def calculate_vehicle_density(vehicle_count, coefficients, lanes_area_pixels):
#         total_vehicle_coefficients = sum(coefficients[class_name] * count for class_name, count in vehicle_count.items())
#         vehicle_density = total_vehicle_coefficients/51
#         return vehicle_density

#     image = cv2.imread(image_path)

#     result_image, vehicle_count, vehicle_classes = detect_objects(image)

#     lanes_image, lanes_area_pixels = detect_lanes(image)

#     plt.figure(figsize=(8, 8))
#     plt.imshow(cv2.cvtColor(lanes_image, cv2.COLOR_BGR2RGB))
#     plt.title("Lane Detection")
#     plt.show()

#     vehicle_coefficients = {
#         'car': 1,
#         'bike': 0.7,
#         'truck': 2.5,
#         'bus': 2.5,
#         'motorbike': 0.7,
#         'bicycle': 0.7,
#         'rickshaw': 0.7,
#         'cycle': 0.7
#     }

#     density = calculate_vehicle_density(vehicle_count, vehicle_coefficients, lanes_area_pixels)
#     print(f"Vehicle Density: {density}")

#     vehicle_volume = sum(vehicle_coefficients[class_name] * count for class_name, count in vehicle_count.items())
#     print(f"Vehicle Volume: {vehicle_volume}")

#     print("Vehicle Counts:")
#     for class_name, count in vehicle_count.items():
#         print(f"{class_name}: {count}")

#     print(density)
#     return density, vehicle_count


def analyze_traffic(image_path):
    weights_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.weights"
    config_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.cfg"
    names_path = "C:\\Users\\Shalv Srivastava\\yolov3\\coco.names"

    net = cv2.dnn.readNet(weights_path, config_path)

    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')

    indian_vehicle_classes = ['car', 'bike', 'truck', 'bus', 'motorbike', 'bicycle', 'rickshaw', 'cycle']

    classes.extend(indian_vehicle_classes)

    confidence_thresholds = {
        'car': 0.93,
        'bike': 0.2,
        'truck': 0.7,
        'bus': 0.9,
        'motorbike': 0.4,
        'bicycle': 0.4,
        'rickshaw': 0.3,
        'cycle': 0.3
    }

    def detect_objects(image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers_names)

        vehicle_count = {class_name: 0 for class_name in indian_vehicle_classes}
        vehicle_classes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]

                class_name = classes[class_id]

                if class_name in indian_vehicle_classes and confidence > confidence_thresholds[class_name]:
                   
                    if class_name in indian_vehicle_classes and vehicle_count[class_name] < 30:
                        vehicle_count[class_name] += 1

                    vehicle_classes.append(class_name)
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image, vehicle_count, vehicle_classes

    image = cv2.imread(image_path)
    
    result_image,vehicle_count, vehicle_classes=detect_objects(image)
    old_image = result_image
    
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # cv2.imshow("Result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    def detect_lanes(image, sf=0.01):
        plt.figure(figsize=(image.shape[1]/80, image.shape[0]/80), dpi=80)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edges = cv2.Canny(image_blur, 50, 150)
        height, width = image.shape[:2]
        roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.array(roi_vertices)], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

        if lines is not None:
            lanes_image = np.zeros_like(image)
            lane_dimensions = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lanes_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lane_dimensions.append(((x1, y1), (x2, y2)))

            lanes_area_pixels = sum((abs(x2 - x1) * abs(y2 - y1)) for (x1, y1), (x2, y2) in lane_dimensions)
            total_area_input_image = height * width
            real_area = (lanes_area_pixels * sf**2) / (lanes_area_pixels / total_area_input_image)

            width = (lanes_area_pixels / 2.5) ** 0.5

            print(f"Total Lane Area:")
            print(f"In Square Pixels: {lanes_area_pixels} pixels^2")
            print(f"Real Area: {real_area} units^2")
            print(f"Calculated Width: {width} units")

            return lanes_image, lanes_area_pixels

        else:
            lanes_image = image.copy()
            return lanes_image, 0

    def calculate_vehicle_density(vehicle_count, coefficients, lanes_area_pixels):
        total_vehicle_coefficients = sum(coefficients[class_name] * count for class_name, count in vehicle_count.items())
        vehicle_density = total_vehicle_coefficients/51
        return vehicle_density

    image = cv2.imread(image_path)

    result_image, vehicle_count, vehicle_classes = detect_objects(image)

    lanes_image, lanes_area_pixels = detect_lanes(image)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(cv2.cvtColor(lanes_image, cv2.COLOR_BGR2RGB))
    # plt.title("Lane Detection")
    # # plt.show()

    vehicle_coefficients = {
        'car': 1,
        'bike': 0.7,
        'truck': 2.5,
        'bus': 2.5,
        'motorbike': 0.7,
        'bicycle': 0.7,
        'rickshaw': 0.7,
        'cycle': 0.7
    }

    density = calculate_vehicle_density(vehicle_count, vehicle_coefficients, lanes_area_pixels)*1.2
    print(f"Vehicle Density: {density}")

    vehicle_volume = sum(vehicle_coefficients[class_name] * count for class_name, count in vehicle_count.items())
    print(f"Vehicle Volume: {vehicle_volume}")

    print("Vehicle Counts:")
    for class_name, count in vehicle_count.items():
        print(f"{class_name}: {count}")

    print(density)
    return density, vehicle_count,result_image

# def calculate_green_light_time(vehicle_count):
#     max_green_light_time = 30

#     # Your custom calculation for vehicle density
#     total_vehicle_coefficients = sum(vehicle_count.values())
#     density = total_vehicle_coefficients / len(vehicle_count)

#     green_light_time = density * max_green_light_time
#     green_light_time = min(green_light_time, max_green_light_time)
#     return green_light_time
def calculate_green_light_time(density):
    max_green_light_time = 30
    green_light_time = density * max_green_light_time * 1.3
    green_light_time = min(green_light_time, max_green_light_time)
    return green_light_time



# In[ ]:




