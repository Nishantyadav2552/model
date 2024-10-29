from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the YOLOv8 model
model = YOLO("D:\\zara\\trybird\\detect\\train42\\weights\\best.pt")
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = data['image']
    image = base64.b64decode(image_data)

    # Convert the image to an array
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference
    results = model.predict(frame)

    # Process results and draw bounding boxes
    bird_detected = False
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            conf = bbox.conf[0]
            cls = int(bbox.cls[0])
            label = model.names[cls]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if label.lower() == 'bird':
                bird_detected = True

    # Encode the frame back to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': result_image, 'bird_detected': bird_detected})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
