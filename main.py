import cv2
import time

# Load Haar Cascade for full-body detection
pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Open CCTV video
capture = cv2.VideoCapture("cctv.mp4")

# If the video file is not found, use the webcam
if not capture.isOpened():
    capture = cv2.VideoCapture(0)

while capture.isOpened():
    time.sleep(0.05)
    ret, frame = capture.read()

    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians
    pedestrian_detected = pedestrian_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3
    )

    # Draw rectangles and labels around detected people
    for (x, y, w, h) in pedestrian_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (224, 255, 255), 3)
        cv2.putText(
            frame,
            "person",
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (224, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Show output
    cv2.imshow("pedestrian detection", frame)

    # Press Esc to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
