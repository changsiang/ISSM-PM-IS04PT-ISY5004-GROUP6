import cv2
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,  required=True, default="reg", help="patient_registration or patient_examination")
    args = parser.parse_args()
    if (args.mode == "reg"):
        run_patient_registration_system()
    elif (args.mode == "exam"):
        run_patient_examination_system()
    else:
        print("Invalid mode")

def run_patient_registration_system(live_display=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_patient_examination_system(live_display=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

run()