import cv2

class PersonDetectionModel:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('model/weights/haarcascade_frontalface_default.xml')
    
    def run(self):
        pass

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 3)
        return faces
    
    def draw(self, frame, faces, color=(0, 255, 0)):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + w), color, 2)
        return frame
    
    def detect_person_count(self, frame):
        faces = self.detect(frame)
        return len(faces)

if __name__ == "__main__":
    app = PersonDetectionModel()