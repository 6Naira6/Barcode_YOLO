from ultralytics import YOLO
import cv2
import os
from pyzbar import pyzbar

model = YOLO("best.pt")
results = model.predict("img4.jpg")
img = cv2.imread("img4.jpg")

# Define a directory to save the cropped objects
output_dir = "output_objects_str"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for r in results:
    boxes = r.boxes
    for i, box in enumerate(boxes):
        b = box.xyxy[0]  # Get box coordinates in (top, left, bottom, right) format

        # Crop the object from the image
        cropped_object = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

        # Save the cropped object as an image in the output directory
        output_filename = os.path.join(output_dir, f"object_{i}.jpg")
        cv2.imwrite(output_filename, cropped_object)

        # Detect barcodes in the cropped object
        image_gray = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(image_gray)

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            if len(barcode_data) == 15:
                print(f"Detected 15-digit barcode: {barcode_data}")

cv2.imshow('YOLO V8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
