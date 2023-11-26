# Adaptive_Traffic_Signal



This code showcases my attempt to leverage computer vision techniques for traffic analysis in images. At its core, the script employs the YOLOv3 (You Only Look Once version 3) object detection model to identify various vehicles in the input image. The YOLOv3 model is pretrained on the COCO dataset and extended with additional classes relevant to Indian traffic, including cars, bikes, trucks, buses, and more.

Upon calling the analyze_traffic function with an image file path, the script initiates object detection, analyzes traffic conditions, and computes metrics such as vehicle density, volume, and counts for each vehicle class. Additionally, the script integrates lane detection using Canny edge detection and Hough transform techniques. The detected lanes are visually represented on the image, and their areas are computed both in pixels and real-world units.

Furthermore, I've incorporated a calculation for the green light time at a traffic signal based on the vehicle density, considering different coefficients for various vehicle types and a maximum green light time. The script produces outputs including the calculated vehicle density, vehicle count, and an image featuring bounding boxes around detected vehicles and highlighted lanes.
