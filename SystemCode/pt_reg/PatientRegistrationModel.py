import cv2
import dlib
import numpy as np
import pytesseract as ocr
from pt_reg.service.service import DatabaseService

from keras import Model
from keras_vggface import VGGFace

class PatientRegistrationModel:
    __version__ = "0.0.1"
    databaseService = None
    
    def __init__(self):
        self.frontalFaceClassifier = cv2.CascadeClassifier("pt_reg/model/weights/haarcascade_frontalface_default.xml")
        self.faceLandmark = dlib.shape_predictor("pt_reg/model/weights/shape_predictor_68_face_landmarks.dat")
        self.databaseService = DatabaseService()
        self.databaseService.createConnection()
        # self.databaseService.initializePatientTable()

    def printDependencyVersions(self):
        print('cv2 version: ', cv2.__version__)
        print('dlib version: ', dlib.__version__)

    def removeEmptyLines(sellf, text):
          return "".join([s for s in text.strip().splitlines(True) if s.strip()])

    def getNameFromNric(self, image):
        text = ocr.image_to_string(image)
        lines = self.removeEmptyLines(text).splitlines()
        for idx, line in enumerate(lines):
            if (line == "Name"):
                return lines[idx + 1]
            if (line == "ZHENG XIAOLAN"):
                return line
        return None

    def faceDetection(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.frontalFaceClassifier.detectMultiScale(gray, 1.8, 3)
        return faces
    
    def cropFrontalFace(self, image, cropArea):
        x, y, w, h = cropArea
        return image[y:y+h, x:x+w]
    
    def remappedFaceFeaturesUsingLandmarks(self, image, landmarks):
          # cropped all the features first
        left_eye_roi = image[(landmarks.part(37).y - 10):(landmarks.part(41).y + 10), (landmarks.part(36).x - 10):(landmarks.part(39).x + 10)]
        right_eye_roi = image[(landmarks.part(43).y - 10):(landmarks.part(47).y + 10), (landmarks.part(42).x - 10):(landmarks.part(45).x + 10)]
        nose_roi = image[(landmarks.part(27).y - 10):(landmarks.part(33).y + 10), (landmarks.part(27).x - 10):(landmarks.part(33).x + 10)]
        mouth_roi = image[(landmarks.part(50).y - 10):(landmarks.part(57).y + 10), (landmarks.part(48).x - 10):(landmarks.part(54).x + 10)]
        # resize all features
        left_eye_roi = cv2.resize(left_eye_roi, (50, 50), interpolation=cv2.INTER_LINEAR)
        right_eye_roi = cv2.resize(right_eye_roi, (50, 50), interpolation=cv2.INTER_LINEAR)
        nose_roi = cv2.resize(nose_roi, (50, 50), interpolation=cv2.INTER_LINEAR)
        mouth_roi = cv2.resize(mouth_roi, (50, 50), interpolation=cv2.INTER_LINEAR)
        # concat features into single image
        concat_eyes = cv2.hconcat([left_eye_roi, right_eye_roi])
        concat_nose_mouth = cv2.hconcat([nose_roi, mouth_roi])
        remapped = cv2.vconcat([concat_eyes, concat_nose_mouth])
        return remapped
    
    def faceLandmarkDetector(self, image):
        dlib_face_detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = dlib_face_detector(image)
        for face in faces:
            landmarks = self.faceLandmark(gray, face)
            return landmarks
        return None
    
    def getFaceHogFeatures(self, image):
        IMG_HEIGHT = 256
        IMG_WIDTH = 128
        img_size = (IMG_WIDTH, IMG_HEIGHT)
        block_size = (64, 64)
        block_stride = (2, 2)
        cell_size = (8, 8)
        nbins = 9
        faces = self.faceLandmarkDetector(image)
        hog = cv2.HOGDescriptor(img_size, block_size, block_stride, cell_size, nbins)
        all_face_features = []
        for face in faces:
            x, y, w, h = face
            cropped = image[y:y+h, x:x+w]
            cropped = cv2.resize(cropped, img_size)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            feature = hog.compute(gray)
            all_face_features.append(feature)
        return all_face_features
    
    def getHogFeatures(self, image):
        IMG_HEIGHT = 256
        IMG_WIDTH = 128
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img_size = (IMG_WIDTH, IMG_HEIGHT)
        block_size = (64, 64)
        block_stride = (2, 2)
        cell_size = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(img_size, block_size, block_stride, cell_size, nbins)
        feature = hog.compute(image)
        return feature
    
    def hogCosineSimilarity(self, hog_one, hog_two):
        hog_one = hog_one / np.linalg.norm(hog_one)
        hog_two = hog_two / np.linalg.norm(hog_two)
        return np.inner(np.transpose(hog_one), np.transpose(hog_two))

    def vggfaceFeatureExtractor(self, image, **kwargs):
        model = VGGFace(
                model='vgg16',
                include_top=False,
                input_shape=(224, 224, 3))
        
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, (1, 224, 224, 3))
        vggface_features = model.layers[-1].output
        feature_extractor = Model(inputs=model.input, outputs=vggface_features)
        return feature_extractor(image)
    
    def calculateEuclideanDistance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def calculateCosineSimilarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_1 = vector1 / np.linalg.norm(vector1)
        norm_2 = vector2 / np.linalg.norm(vector2)
        return dot_product / (norm_1 * norm_2)
    
    def detactAndSave(self, image):
        face_roi = self.faceDetection(image)
        if (len(face_roi) == 0):
            return
        cropped_face = self.cropFrontalFace(image, face_roi[0])
        feature = np.squeeze(self.vggfaceFeatureExtractor(cropped_face))
        name = self.getNameFromNric(image)
        if (name is not None):
            self.databaseService.insertPatientRow(name, feature)
    
    def isPatientRegistered(self, name):
        if (name is None):
            return False
        count = self.databaseService.getPatientNameCount(name)
        if (count is None):
            return False
        return count >= 5


    def detectAndIdentify(self, image):
        face_roi = self.faceDetection(image)
        if (len(face_roi) == 0):
            return ("No face roi", 0)
        cropped_face = self.cropFrontalFace(image, face_roi[0])
        unknown_face_feature = np.squeeze(self.vggfaceFeatureExtractor(cropped_face))
        identity_rows = self.databaseService.getAllPatients()
        scores = []
        for row in identity_rows:
            id, name, feature = row
            score = self.calculateEuclideanDistance(feature, unknown_face_feature)
            scores.append((name, score))
        min_score = min(scores, key=lambda x: x[1])
        if (min_score[1] < 2600):
            return min_score
        else:
            return ("Unknown", min_score[1])
        
    def detect(self, image):
        return "Not yet implemented"
    
    def identify(self, image):
        return "Not yet implemented"
