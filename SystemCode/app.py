import cv2
import argparse
from SystemCode.pt_reg.register import PatientRegistrationModel

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,  required=True, default="reg", help="patient_registration or patient_examination")
    args = parser.parse_args()
    if (args.mode == "reg"):
        run_patient_registration_system()
    elif (args.mode == "exam"):
        run_patient_examination_system()
    elif (args.mode == "identify"):
        run_patient_identification_system()
    else:
        print("Invalid mode")

def run_patient_registration_system(live_display=True):
    reg_system = PatientRegistrationModel()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pt_name = reg_system.getNameFromNric(frame)
        reg_system.detactAndSave(frame)
        is_registered = reg_system.isPatientRegistered(pt_name)
        if (is_registered):
            print("Patient {} is registered".format(pt_name))
        else:
            print("Patient {} is not registered".format(pt_name))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def run_patient_identification_system(live_display=True):
    reg_system = PatientRegistrationModel()
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