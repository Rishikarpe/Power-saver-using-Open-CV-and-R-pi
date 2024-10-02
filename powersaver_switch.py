import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Setup GPIO for controlling relay (assuming pin 17)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

# Load YOLOv4 model
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load YOLO model
try:
    net, output_layers = load_yolo()
except cv2.error as e:
    print("Error loading YOLO model:", e)
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:  # Only consider persons
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Person count based on detection
    person_count = len(indexes) if indexes is not None else 0
    
    # Determine electricity state based on person count
    if person_count > 0:
        electricity_status = "Electricity ON"
        GPIO.output(17, GPIO.HIGH)  # Turn ON the relay (assumed active HIGH)
    else:
        electricity_status = "Electricity OFF"
        GPIO.output(17, GPIO.LOW)  # Turn OFF the relay (assumed active LOW)

    # Draw bounding boxes around detected persons
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the current person count on the frame
    cv2.putText(frame, f'Persons: {person_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display electricity status on the frame
    cv2.putText(frame, electricity_status, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv4 Person Counter", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Clean up GPIO on exit
GPIO.cleanup()
