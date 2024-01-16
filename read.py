import cv2

# Threshold to detect objects
thres = 0.6 # Adjust the threshold for your specific requirements

# Open the webcam (camera index 0)
cap = cv2.VideoCapture(0)
cap.set(3, 2000)  # Increase the resolution for better image quality
cap.set(4, 1900)

# Load class names from a file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to model configuration and weights
config_Path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_Path = 'frozen_inference_graph.pb'

# Load the detection model
net = cv2.dnn_DetectionModel(weights_Path, config_Path)
net.setInputSize(310, 310)  # Adjust input size
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    
    # Detect objects using the loaded model
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.3)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw bounding box and label
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # Display the image with detection results
    cv2.imshow("Object Detection", img)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
