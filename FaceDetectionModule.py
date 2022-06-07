import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_conf=0.5, complexity=1):
        self.min_detection_conf = min_detection_conf
        self.complexity = complexity

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_conf)

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        #print(results)
        bounding_boxes = []

        if self.results.detections:
            for face_id, detection in enumerate(self.results.detections):
                class_bounding_box = detection.location_data.relative_bounding_box
                h, w, ch = img.shape
                bounding_box = int(class_bounding_box.xmin * w), int(class_bounding_box.ymin * h), \
                               int(class_bounding_box.width * w), int(class_bounding_box.height * h)
                bounding_boxes.append([face_id, bounding_box, detection.score])
                if draw:
                    img = self.fancy_draw(img, bounding_box)

                    cv2.putText(
                        img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
                    )
        return img, bounding_boxes

    def fancy_draw(self, img, bounding_box, l=30, thickness=5, rect_thickness=1):
        # Origin
        x, y, w, h = bounding_box
        # Bottom right corner
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bounding_box, (255, 0, 255), rect_thickness)
        # Top left: x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), thickness)
        # Top right: x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), thickness)
        # Bottom left: x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), thickness)
        # Bottom right: x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), thickness)
        return img

def main():
    # Run the video
    capture = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceDetector()

    while True:
        success, img = capture.read()
        img, bounding_boxes = detector.find_faces(img)
        print(bounding_boxes)
        # Get the fps
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, f'{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()