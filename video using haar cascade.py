import cv2
# Load the pre-trained Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'https://github.com/spmallick/mallick_cascades/blob/master/haarcascades/haarcascade_russian_plate_number.xml')
# Open video capture (0 for default webcam, or provide video filepath)
cap = cv2.VideoCapture(0)
while True:
 ret, frame = cap.read()
 if not ret:
     break
 # Convert the frame to grayscale
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 # Detect license plates
 plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1,
minNeighbors=5, minSize=(60, 60))
 # Draw rectangles around the detected license plates
 for (x, y, w, h) in plates:
     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 # Show the resulting frame
 cv2.imshow('License Plate Detection', frame)
 # Exit when the user presses 'q'
 if cv2.waitKey(1) & 0xFF == ord('q'):
     break
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
