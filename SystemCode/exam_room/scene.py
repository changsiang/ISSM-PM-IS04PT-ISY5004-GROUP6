import cv2
import torch
import os
import pickle
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import dlib
import numpy as np

class SceneRecongitionModel:
    def __init__(self) -> None:
        self.action_classifier = pickle.load(open('exam_room/model/weights/svm_model.pkl', 'rb'))
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("exam_room/model/weights/shape_predictor_68_face_landmarks.dat")
        self.class_labels = ['others', 'right', 'left', 'left_pinhole', 'right_pinhole']
        self.checkpoint_path = os.path.join("sam_vit_h_4b8939.pth")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_type = "vit_h"
        pass

    def run(self):
        pass
    
    # roi should be a tuple of (x, y, w, h)
    def predict(self, frame, roi):
        # input is frame and output is prediction of action
        # this method will use the single frame approach
        thr = self.generate_segmentation_masking_using_thresholding(frame, roi)
        thr_resize = cv2.resize(thr, (128, 128), interpolation=cv2.INTER_LINEAR)
        pred = self.action_classifier.predict(thr_resize.reshape(-1, 128 * 128))
        return pred[0]
    
    def class_id_to_name(self, class_id):
        return self.class_labels[class_id]
    
    def draw_label(self, frame, label, roi):
        cv2.putText(frame, label, (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
        return frame
    
    def generate_segmentation_masks_using_sam(self, frame, roi=None):
        sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        target_image = frame
        if (roi != None):
            target_image = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        sam_result = mask_generator.generate(target_image)
        masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)]
        return masks
    
    def generate_segmentation_masking_using_thresholding(self, frame, roi=None):
        target_image = frame
        if (roi != None):
            target_image = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(target_image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thr
    
    def detect_glasses(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        if (len(faces) > 1 or len(faces) == 0):
            return False
        left = faces[0].left()
        top = faces[0].top()
        width = faces[0].right() - left
        height = faces[0].bottom() - top
        roi = gray[top+10:top+height-100, left+30:left+width-20]
        if (len(roi) == 0):
            return False
        blur = cv2.GaussianBlur(np.array(roi), (3, 3), sigmaX=1.7, sigmaY=1.7)
        edges = cv2.Canny(blur, 100, 200)
        edges_center = edges.T[(int(len(edges.T)/2))]
        if 255 in edges_center:
            return True
        return False

if __name__ == "__main__":
    app = SceneRecongitionModel()