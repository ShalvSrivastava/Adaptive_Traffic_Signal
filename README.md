# Adaptive_Traffic_Signal

**Traffic Analysis**
This code showcases my attempt to leverage computer vision techniques for traffic analysis in images. At its core, the script employs the YOLOv3 (You Only Look Once version 3) object detection model to identify various vehicles in the input image. The YOLOv3 model is pre-trained on the COCO dataset and extended with additional classes relevant to Indian traffic, including cars, bikes, trucks, buses, and more.

Upon calling the analyze_traffic function with an image file path, the script initiates object detection, analyzes traffic conditions, and computes metrics such as vehicle density, volume, and counts for each vehicle class. Additionally, the script integrates lane detection using Canny edge detection and Hough transform techniques. The detected lanes are visually represented on the image, and their areas are computed both in pixels and real-world units.

Furthermore, I've incorporated a calculation for the green light time at a traffic signal based on the vehicle density, considering different coefficients for various vehicle types and a maximum green light time. The script produces outputs including the calculated vehicle density, vehicle count, and an image featuring bounding boxes around detected vehicles and highlighted lanes.


**Video Analysis**

This Python code utilizes the YOLOv3 (You Only Look Once) object detection algorithm to count the number of cars in a given video. The YOLOv3 model is pre-trained on the COCO (Common Objects in Context) dataset, enabling it to detect various objects, including cars. The video processing is performed using the OpenCV library.

The code first loads the YOLOv3 model with its weights, configuration, and class names. It then reads the input video and sets up parameters such as frame skipping, resizing, and detection thresholds. The video frames are iterated through, and for every nth frame (determined by the skip_frames variable), the YOLOv3 model is applied to detect objects.

Detected objects are filtered based on their class ID (2 corresponds to a car in the COCO dataset) and a confidence threshold. The confidence threshold ensures that only high-confidence car detections are considered. The code accumulates the count of detected cars throughout the video.

Finally, the results are returned, including the total count of cars and the detailed count for each frame. The example usage at the end demonstrates how to apply the function to a specific video file and prints the total car count.
