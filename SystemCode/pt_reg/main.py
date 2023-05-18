import cv2
from model.PatientRegistrationModel import PatientRegistrationModel
from service.service import DatabaseService

# main thread that runs the data capture session
## 1. run 2 cameras concurrently (2 threads)
## 2. camera 0 for real person, camera 1 for id 
## 3. camera 1 identify the presence of photo id
### 3.1. consider race condition. Camera 1 will identify photo id data and hold the data temporarily
## 4. once photo id is identify, capture photo id data
## 5. person will present to camera 0
## 6. check the similarity between id and person
## 7. If same, register
## 8. If different clarify

# Set up video capture
cap = cv2.VideoCapture(0)

# Define the lower and upper bounds of the pink color in HSV color space
lower_pink = (150, 50, 50)
upper_pink = (170, 255, 255)

reg_model = PatientRegistrationModel()

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    faces = reg_model.faceDetection(frame)
    # name = reg_model.getNameFromNric(frame)
    name = reg_model.detectAndIdentify(frame)
    
    if (len(faces) > 0):
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)
        if (name is not None):
            cv2.putText(frame, "{}, {}".format(name[0], name[1]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
            
    # # Convert the frame to HSV color space
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # # Threshold the frame to extract pixels within the pink color range
    # mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # # Find contours of the pink regions in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Check if any contours are present and draw a bounding box around the largest one    
    # if len(contours) > 0:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, 'Pink card detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show the frame with the bounding box if a pink card is detected
    cv2.imshow('Pink card detection', frame)
    # reg_model.detactAndSave(frame)
    
    # Wait for a key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
