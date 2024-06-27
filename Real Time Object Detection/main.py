import cv2
import numpy as np


# Load YOLO network
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getUnconnectedOutLayersNames()

# Define custom classes and search-related variables
custom_classes = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV Monitor", "Laptop", "Mouse", "Remote", "Keyboard", "Cellphone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Dryer", "Toothbrush"]
search_term = ""
search_bar_active = False
detected_objects = []
search_bar_color = (255, 255, 255)

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # Process the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels for the filtered detections
    for i in indices:

        x, y, w, h = boxes[i]
        label = f"{custom_classes[class_ids[i]]}: {confidences[i]:.2f}"

        if search_term.lower() in custom_classes[class_ids[i]].lower():
            color = (0, 255, 0)  # Green for matching object
        else:
            color = (0, 0, 255)  # Red for non-matching objects

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the search bar
    cv2.rectangle(frame, (10, 10), (210, 40), (255, 255, 255), -1)
    cv2.putText(frame, "Search:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, search_term, (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 8:  # Backspace key
        search_term = search_term[:-1]
    elif key == 13:  # Enter key
        search_term = ""
    elif key != 255:
        search_term += chr(key)

cap.release()
cv2.destroyAllWindows()
