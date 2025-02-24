from ultralytics import YOLO
import cv2
from pyzbar import pyzbar
import os

# Initialize YOLO model
model = YOLO("best.pt")

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use the default camera (0) or specify the camera index

# Create a directory to save the detected objects
output_dir = "output_objects_live"
os.makedirs(output_dir, exist_ok=True)

detected_barcodes = set()

#cap = cv2.VideoCapture('rtsp://192.168.111.132:8080/h264_pcm.sdp')    # Reading from camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Display the frame
    cv2.imshow('YOLO V8 Detection', frame)

    key = cv2.waitKey(1)

    if key & 0xFF == 27:  # Press 'Esc' key to exit
        break
    elif key & 0xFF == 32:  # Press 'Space' key to capture and process a frame
        # Perform object detection using YOLO on the frame
        results = model.predict(frame)

        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                b = box.xyxy[0]  # Get box coordinates in (top, left, bottom, right) format

                # Crop the object from the frame
                cropped_object = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

                if cropped_object.size > 0:
                    # Save the cropped object as an image in the output directory
                    output_filename = os.path.join(output_dir, f"object_{i}.jpg")
                    cv2.imwrite(output_filename, cropped_object)

                    # Detect barcodes in the cropped object
                    image_gray = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)
                    barcodes = pyzbar.decode(image_gray)

                    for barcode in barcodes:
                        barcode_data = barcode.data.decode('utf-8')
                        if len(barcode_data) == 15 and (barcode_data.startswith('35') or barcode_data.startswith('86')) and barcode_data not in detected_barcodes:
                            detected_barcodes.add(barcode_data)
                            print(f"Detected 15-digit barcode: {barcode_data}")

        print("Frame processed.")  # Print a message after each frame is processed

# Output detected barcodes to a text file
with open("detected_barcodes.txt", "w") as txt_file:
    for barcode in detected_barcodes:
        txt_file.write(barcode + '\n')

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
