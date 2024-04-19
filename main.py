from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('football.pt')

# Load video
video_path = 'g1.mp4'
cap = cv2.VideoCapture(video_path)

# Read frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster inference (optional)
    # frame = cv2.resize(frame, (new_width, new_height))

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Plot results
    frame_ = results[0].plot()

    # Visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()