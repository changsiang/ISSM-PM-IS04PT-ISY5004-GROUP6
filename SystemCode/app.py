import cv2
import argparse
from pt_reg.PatientRegistrationModel import PatientRegistrationModel
from exam_room.scene import SceneRecongitionModel

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
    elif (args.mode == "init"):
        init_pt_database()
    else:
        print("Invalid mode")

def init_pt_database():
    reg_system = PatientRegistrationModel()
    reg_system.databaseService.initializePatientTable()

def run_patient_registration_system(live_display=True):
    reg_system = PatientRegistrationModel()
    # Create the VideoWriter object
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pt_name = reg_system.getNameFromNric(frame)
        reg_system.detactAndSave(frame)
        is_registered = reg_system.isPatientRegistered(pt_name)
        if (is_registered is False):
            cv2.putText(frame, "Please scan NRIC to register", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Patient: {} already registered with system".format(pt_name), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

def run_patient_identification_system(live_display=True):
    reg_system = PatientRegistrationModel()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = reg_system.faceDetection(frame)
        if (len(faces) > 1):
            cv2.putText(frame, "Error: More than 1 person is present in frame", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
        elif (len(faces) == 1):
            pt_ident, score = reg_system.detectAndIdentify(frame)
            x, y, w,h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Patient: {}-{}".format(pt_ident, score), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_patient_examination_system(live_display=True):
    exam_system = SceneRecongitionModel()
    cap = cv2.VideoCapture('demo_video/IMG_3682.mp4')
    output = cv2.VideoWriter('IMG_3682.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(3)),int(cap.get(4))))
    current_patient = "Unknown"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = exam_system.face_detector(frame, 1)
        cv2.putText(frame, "Current Patient: {}".format(current_patient), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
        if (len(faces) == 1):
            x = faces[0].left()
            y = faces[0].top()
            w = faces[0].right() - x
            h = faces[0].bottom() - y
            current_patient = "ZHENG XIAOLAN"
            hasGlasses = exam_system.detect_glasses(frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if (hasGlasses):
                cv2.putText(frame, "Glasses Present", (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
        action = exam_system.predict(frame, (790, 340, 159, 129))
        action_lbl = exam_system.class_id_to_name(action)
        if (action is not None):
            cv2.putText(frame, "Current Action: {}".format(action_lbl), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        # cv2.imshow('frame', frame)
        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()

run()