import cv2

cap = cv2.VideoCapture('demo_video/IMG_3682.mp4')
count = 0
while True:
    print('countL: ', count)
    ret, frame = cap.read()
    if not ret:
        break
    action_lbl = "none"
    if count > 520 and count < 680:
        action_lbl = "left"
    if count > 700 and count < 820:
        action_lbl = "right"
    cv2.putText(frame, "Current Action: {}".format(action_lbl), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(frame, (790, 340), (790+129, 340+159), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1
cap.release()
cv2.destroyAllWindows()


