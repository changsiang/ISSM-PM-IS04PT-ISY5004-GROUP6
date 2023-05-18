import cv2
import scene
import supervision as sv
import torch
import os
import numpy as np
import time
import pickle
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class SceneRecongitionModel:
    def __init__(self) -> None:
        self.action_classifier = pickle.load(open('model/weights/svm_model.pkl', 'rb'))
        self.class_labels = ['invalid_action', 'right', 'left', 'left_pinhole', 'right_pinhole']
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_roi = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        thr = cv2.threshold(frame_roi, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
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

if __name__ == "__main__":
    app = SceneRecongitionModel()