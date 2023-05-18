import cv2
import person
import scene
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import torch
import os
import numpy as np
import time
import pickle

CHECKPOINT_PATH = os.path.join("sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

def main():
    print(cv2.__version__)
    person.person_count()
    person.person_identifier()
    scene.identify()

def calculate_eluclidean(vec1, vec2):
    # calculate euclidean distance between two vectors
    return np.linalg.norm(vec1 - vec2)

def test_sam():
    mask_annotator = sv.MaskAnnotator()
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    face_model = cv2.CascadeClassifier("model/weights/haarcascade_frontalface_default.xml")
    action_classifier = pickle.load(open('model/weights/svm_model.pkl', 'rb'))
    lbls = ['invalid_action', 'right', 'left', 'left_pinhole', 'right_pinhole']
    # using cv2 to read mp4 video file
    cap = cv2.VideoCapture('demo/demo_video.mp4')
    output = cv2.VideoWriter('demo/demo_video_output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (848, 480))
    hasFace = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('frame', frame)
        faces = face_model.detectMultiScale(frame, 1.2, 3)
        if (len(faces) > 0):
            hasFace = True
            x, y, w, h = faces[0]
            # print(faces[0])
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)

        # [392 138 109 109]
        if (hasFace):
            cropped_image = frame[138:138+109, 392:392+109]
            sam_result = mask_generator.generate(cropped_image)
            # detections = sv.Detections.from_sam(sam_result=sam_result)
            masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)]
            for mask in masks:
                img = np.array(mask * 255, dtype=np.uint8)
                thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
                mask = cv2.resize(thr, (128, 128), interpolation=cv2.INTER_LINEAR)
                pred = action_classifier.predict(mask.reshape(-1, 128 * 128))
                if (pred[0] > 1):
                    print(lbls[pred[0]])
                    cv2.putText(frame, lbls[pred[0]], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
            # annotated_image = mask_annotator.annotate(scene=cropped_image.copy(), detections=detections)
            # frame[138:138+109, 392:392+109] = annotated_image
        
        # sam_result = mask_generator.generate(frame)
        # detections = sv.Detections.from_sam(sam_result=sam_result)
        # annotated_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        # masks = [
        #     mask['segmentation'] for mask 
        #     in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        # ]
        #cv2 write frames to video
        output.write(frame)
        # cv2.imshow('mask', frame)       
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video capture and close all windows
    cap.release()
    output.release()
    cv2.destroyAllWindows()


def test_thr():
    mask_annotator = sv.MaskAnnotator()
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    face_model = cv2.CascadeClassifier("model/weights/haarcascade_frontalface_default.xml")
    action_classifier = pickle.load(open('model/weights/svm_model.pkl', 'rb'))
    lbls = ['invalid_action', 'right', 'left', 'left_pinhole', 'right_pinhole']
    # using cv2 to read mp4 video file
    cap = cv2.VideoCapture('demo/demo_video.mp4')
    output = cv2.VideoWriter('demo/demo_video_output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (848, 480))
    hasFace = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('frame', frame)
        faces = face_model.detectMultiScale(frame, 1.2, 3)
        if (len(faces) > 0):
            hasFace = True
            x, y, w, h = faces[0]
            # print(faces[0])
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)

        # [392 138 109 109]
        # if (hasFace):
        cropped_image = frame[120:120+256, 350:350+256]
        thr = cv2.threshold(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), 250, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thr_resize = cv2.resize(thr, (128, 128), interpolation=cv2.INTER_LINEAR)
        pred = action_classifier.predict(thr_resize.reshape(-1, 128 * 128))
        if (pred[0] > 0):
            print(lbls[pred[0]])
            cv2.putText(frame, lbls[pred[0]], (350, 350 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
        # frame[138:138+256, 392:392+256] = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        
        # sam_result = mask_generator.generate(frame)
        # detections = sv.Detections.from_sam(sam_result=sam_result)
        # annotated_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        # masks = [
        #     mask['segmentation'] for mask 
        #     in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        # ]
        #cv2 write frames to video
        output.write(frame)
        # cv2.imshow('mask', frame)       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    output.release()
    cv2.destroyAllWindows()

# test_sam()

test_thr()