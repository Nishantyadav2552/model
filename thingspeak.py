import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import signal
import requests

# ThingSpeak configuration
THINGSPEAK_API_KEY = 'DHZZY963GDP38OHR'
THINGSPEAK_URL = 'https://api.thingspeak.com/update'

# Load the YOLOv8 model
model = YOLO("D:\\zara\\trybird\\detect\\train42\\weights\\best.pt")

# Function to handle the signal
def signal_handler(sig, frame):
    global running
    print("You pressed Ctrl+C!")
    running = False

# Function to send data to ThingSpeak
def send_to_thingspeak(bird_detected, num_birds):
    data = {
        'api_key': THINGSPEAK_API_KEY,
        'field1': int(bird_detected),
        'field2': num_birds
    }
    response = requests.post(THINGSPEAK_URL, data=data)
    if response.status_code == 200:
        print("Data sent to ThingSpeak successfully.")
    else:
        print(f"Failed to send data to ThingSpeak. Status code: {response.status_code}")

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Path to the video file
video_path = "D:/Bird_detection/bird.mp4"

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
    
    # Check for bird detection and count the number of birds
    bird_detected = False
    num_birds = 0

    # Process results
    for result in results:
        for bbox in result.boxes:
            bird_detected = True  # Bird detected
            num_birds += 1
            x1, y1, x2, y2 = bbox.xyxy[0]
            conf = bbox.conf[0].item()  # Convert tensor to Python float
            cls = bbox.cls[0].item()  # Convert tensor to Python int
            label = model.names[int(cls)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Send detection data to ThingSpeak (0 or 1 for bird detection, number of birds detected)
    send_to_thingspeak(bird_detected, num_birds)
    
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
