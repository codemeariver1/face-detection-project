import cv2
import mediapipe as mp
import time

# Run the video
capture = cv2.VideoCapture(0)
prevTime = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.75)

while True:
    success, img = capture.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    #print(results)

    if results.detections:
        for face_id, detection in enumerate(results.detections):
            #mp_draw.draw_detection(img, detection)
            #print(face_id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            class_bounding_box = detection.location_data.relative_bounding_box
            h, w, ch = img.shape
            bounding_box = int(class_bounding_box.xmin * w), int(class_bounding_box.ymin * h), \
                           int(class_bounding_box.width * w), int(class_bounding_box.height * h)
            cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
            cv2.putText(
                img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
            )


    # Get the fps
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)