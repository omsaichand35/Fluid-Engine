import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config.settings import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
import os

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

class DemoResults:
    def __init__(self, detection_result):
        self.multi_hand_landmarks = []
        for i in range(len(detection_result.hand_landmarks)):
            # Make a dummy class to expose .landmark array
            class HandLms:
                pass
            hlms = HandLms()
            hlms.landmark = detection_result.hand_landmarks[i]
            self.multi_hand_landmarks.append(hlms)

class HandTracker:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        return DemoResults(detection_result)

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    idx1, idx2 = connection
                    lm1 = hand_landmarks.landmark[idx1]
                    lm2 = hand_landmarks.landmark[idx2]
                    x1, y1 = int(lm1.x * w), int(lm1.y * h)
                    x2, y2 = int(lm2.x * w), int(lm2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                # Draw points
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return frame
