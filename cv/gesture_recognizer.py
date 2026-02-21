import time
import math
import sys
from collections import deque
from pathlib import Path

# Ensure project root is on path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv.gesture_types import Gesture

class GestureRecognizer:
    def __init__(self):
        self.wrist_history = deque(maxlen=20)
        self.last_rotation_time = 0

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def fingers_up(self, hand_landmarks):
        up = []
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        
        # Thumb
        if hand_landmarks[tips[0]].x < hand_landmarks[pips[0]].x:
            up.append(1)
        else:
            up.append(0)
            
        # Other fingers
        for i in range(1, 5):
            if hand_landmarks[tips[i]].y < hand_landmarks[pips[i]].y:
                up.append(1)
            else:
                up.append(0)
                
        return up

    def detect_single_hand(self, hand_landmarks):
        up = self.fingers_up(hand_landmarks)
        
        # index & middle pointing up -> DOG
        if up[1] == 1 and up[2] == 1 and up[3] == 0 and up[4] == 0:
            return Gesture.DOG
            
        # All fingers folded -> RASENGAN (close fist)
        if sum(up) == 0:
            return Gesture.RASENGAN
            
        # All fingers open -> FIREBALL or AURA
        if sum(up) >= 4:
            spread = self.get_distance(hand_landmarks[8], hand_landmarks[20])
            if spread > 0.25:
                return Gesture.AURA
            else:
                return Gesture.FIREBALL
                
        return Gesture.UNKNOWN

    def detect_two_hand_seal(self, hand1, hand2):
        wrist1, wrist2 = hand1[0], hand2[0]
        index1, index2 = hand1[8], hand2[8]
        thumb1, thumb2 = hand1[4], hand2[4]
        
        palms_touching = self.get_distance(wrist1, wrist2) < 0.15
        index_fingers_touch = self.get_distance(index1, index2) < 0.1
        thumbs_touch = self.get_distance(thumb1, thumb2) < 0.1
        
        triangle_shape = thumbs_touch and index_fingers_touch and not palms_touching

        if palms_touching:
            return Gesture.RAM
            
        if triangle_shape:
            return Gesture.FIREBALL
            
        # Stub complex ones
        if thumbs_touch and not index_fingers_touch:
            return Gesture.TIGER
            
        return Gesture.UNKNOWN

    def recognize(self, results) -> str:
        """
        Converts mediapipe hand landmarks into a Gesture constant.
        """
        if not results or not results.multi_hand_landmarks:
            return Gesture.NONE
            
        if len(results.multi_hand_landmarks) == 2:
            return self.detect_two_hand_seal(
                results.multi_hand_landmarks[0].landmark,
                results.multi_hand_landmarks[1].landmark
            )
        else:
            return self.detect_single_hand(
                results.multi_hand_landmarks[0].landmark
            )


class GestureSequenceRecognizer:
    def __init__(self, timeout=2.0):
        self.sequence = []
        self.last_gesture_time = time.time()
        self.timeout = timeout
        
    def add_gesture(self, gesture):
        current_time = time.time()
        if current_time - self.last_gesture_time > self.timeout:
            self.sequence.clear()
            
        if gesture != Gesture.NONE and gesture != Gesture.UNKNOWN:
            if not self.sequence or self.sequence[-1] != gesture:
                self.sequence.append(gesture)
                self.last_gesture_time = current_time
                
    def get_sequence(self):
        current_time = time.time()
        if current_time - self.last_gesture_time > self.timeout:
            self.sequence.clear()
        return self.sequence
        
    def clear(self):
        self.sequence.clear()
