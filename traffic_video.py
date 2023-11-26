import cv2
import numpy as np

def process_video(video_path):
    # Paths
    weights_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.weights"
    config_path = "C:\\Users\\Shalv Srivastava\\yolov3\\yolov3.cfg"
    names_path = "C:\\Users\\Shalv Srivastava\\yolov3\\coco.names"

    # Load YOLOv3
    net = cv2.dnn.readNet(weights_path, config_path)

    # Load classes
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load video
    video = cv2.VideoCapture(video_path)

    # Video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Processing settings
    skip_frames = 15
    resize_factor = 0.3
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

    # Object detection and counting
    car_count = 0

    for i in range(0, frame_count, skip_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Extract information from detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Adjust the class_id and confidence threshold according to your needs
                if class_id == 2 and confidence > 0.985:
                    car_count += 1

    # Release video capture object
    video.release()
    cv2.destroyAllWindows()

    # Calculate and return results
    result = round(car_count)
    return result, car_count



# Example usage
video_path = "C:\\Users\\Shalv Srivastava\\To Save Python code\\videoTest.mp4"


total_cars, car_count = process_video(video_path)
#print(f"Total cars detected: {total_cars}")
print(f"Car count: {car_count}")
