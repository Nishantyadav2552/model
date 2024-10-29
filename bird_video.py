import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import signal

# Load the YOLOv8 model
model = YOLO("D:\\zara\\trybird\\detect\\train42\\weights\\best.pt")

# Function to handle the signal
def signal_handler(sig, frame):
    global running
    print("You pressed Ctrl+C!")
    running = False

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Path to the video file
video_path = "D:/Bird_detection/test.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Flag to control the loop
running = True

plt.ion()  # Turn on interactive mode for Matplotlib

while running:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error: Failed to capture image.")
        break
    
    # Perform inference
    results = model.predict(frame)
    
    # Process results
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]
            conf = bbox.conf[0]
            cls = bbox.cls[0]
            label = model.names[int(cls)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the output frame using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axis
    plt.draw()
    plt.pause(0.001)  # Pause to update the figure
    plt.clf()  # Clear the figure for the next frame

# Release the video
cap.release()
plt.ioff()  # Turn off interactive mode
plt.show()  # Ensure the last frame is shown
cv2.destroyAllWindows()
