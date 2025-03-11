import cv2
# Load pre-trained Haar cascades for face and facial features
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_mcs_mouth.xml')
# Open video capture (0 for default webcam, or provide a video file)
cap = cv2.VideoCapture(0)
while True:
 ret, frame = cap.read()
 if not ret:
     break
 # Convert the frame to grayscale
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 # Detect faces in the image
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
minNeighbors=5, minSize=(30, 30))
 # Loop over the faces detected
 for (x, y, w, h) in faces:
 # Draw a rectangle around the face
     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
 # Region of interest (ROI) for eyes, nose, mouth
 roi_gray = gray[y:y + h, x:x + w]
 roi_color = frame[y:y + h, x:x + w]
 # Detect eyes within the face region
 eyes = eye_cascade.detectMultiScale(roi_gray)
 for (ex, ey, ew, eh) in eyes:
     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
 # Detect nose within the face region
 nose = nose_cascade.detectMultiScale(roi_gray)
 for (nx, ny, nw, nh) in nose:
     cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
 # Detect mouth within the face region
 mouth = mouth_cascade.detectMultiScale(roi_gray)
 for (mx, my, mw, mh) in mouth:
     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255,
255), 2)
 # Display the resulting frame
 cv2.imshow('Facial Features Detection', frame)
 # Exit when 'q' is pressed
 if cv2.waitKey(1) & 0xFF == ord('q'):
     break
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
 
