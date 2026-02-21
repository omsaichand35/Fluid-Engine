import time
import math
import sys
from pathlib import Path
from collections import deque

# Ensure project root is on path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv.gesture_types import GestureType

class GestureRecognizer:
    def __init__(self):
        # Store a history of wrist positions to detect rotation
        self.wrist_history = deque(maxlen=20)
        self.last_rotation_time = 0

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def is_circular_motion(self, current_time):
        """
        Check if the wrist has moved in a circular path recently
        """
        if len(self.wrist_history) < 15:
            return False
            
        if current_time - self.last_rotation_time < 1.0:
            return False

        # Calculate bounding box of wrist movement
        min_x = min(p.x for p in self.wrist_history)
        max_x = max(p.x for p in self.wrist_history)
        min_y = min(p.y for p in self.wrist_history)
        max_y = max(p.y for p in self.wrist_history)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate total distance traveled vs straight line distance
        path_length = sum(self.get_distance(self.wrist_history[i], self.wrist_history[i-1]) 
                          for i in range(1, len(self.wrist_history)))
        direct_distance = self.get_distance(self.wrist_history[0], self.wrist_history[-1])
        
        # A circle has path_length roughly pi * diameter. 
        # Path length should be much larger than direct distance for circular or oscillating motion.
        # Also need significant movement (width and height > 0.05)
        if width > 0.05 and height > 0.05 and path_length > 2.0 * direct_distance and path_length > 0.3:
            self.last_rotation_time = current_time
            self.wrist_history.clear()
            return True
            
        return False

    def recognize(self, results) -> GestureType:
        """
        Converts mediapipe hand landmarks into a GestureType.
        Implementation of Naruto Hand Signs for Jutsus.
        """
        if not results.multi_hand_landmarks:
            self.wrist_history.clear()
            return GestureType.NONE

        hand_landmarks = results.multi_hand_landmarks[0].landmark
        
        # Track wrist history for Rotation (Wind Tornado)
        current_time = time.time()
        wrist = hand_landmarks[0]
        self.wrist_history.append(wrist)
        
        # Check motion-based gesture first
        if self.is_circular_motion(current_time):
            return GestureType.WIND

        # Key Landmarks Reference
        # Thumb: Tip 4, PIP 3
        # Index: Tip 8, PIP 6
        # Middle: Tip 12, PIP 10
        # Ring: Tip 16, PIP 14
        # Pinky: Tip 20, PIP 18

        index_tip, index_pip = hand_landmarks[8], hand_landmarks[6]
        middle_tip, middle_pip = hand_landmarks[12], hand_landmarks[10]
        ring_tip, ring_pip = hand_landmarks[16], hand_landmarks[14]
        pinky_tip, pinky_pip = hand_landmarks[20], hand_landmarks[18]
        thumb_tip = hand_landmarks[4]

        # Fingers folded state based on y position relative to PIP
        # Note: In MediaPipe, y grows downwards. So tip.y > pip.y means finger is folded DOWN.
        # However, this depends on hand orientation (if pointing up). 
        # We assume standard hand upright pose for these rules.
        index_folded = index_tip.y > index_pip.y
        middle_folded = middle_tip.y > middle_pip.y
        ring_folded = ring_tip.y > ring_pip.y
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        index_open = index_tip.y < index_pip.y
        middle_open = middle_tip.y < middle_pip.y
        ring_open = ring_tip.y < ring_pip.y
        pinky_open = pinky_tip.y < pinky_pip.y

        # 1. Closed Fist (Rasengan)
        if index_folded and middle_folded and ring_folded and pinky_folded:
            # Check thumb is close to palm or other fingers (roughly folded)
            dist_thumb_index = self.get_distance(thumb_tip, index_tip)
            if dist_thumb_index < 0.15:
                return GestureType.RASENGAN

        # 2. Open Palm Forward (Fireball) & 5. Chakra Aura (Wide Open)
        if index_open and middle_open and ring_open and pinky_open:
            # Check spread between index and pinky to differentiate Aura and Fireball
            spread = self.get_distance(index_tip, pinky_tip)
            
            if spread > 0.25: # Wide open
                return GestureType.AURA
            else: # Fingers together but open
                return GestureType.FIREBALL

        # 3. Two Finger Point (Chidori)
        if index_open and middle_open and ring_folded and pinky_folded:
            return GestureType.CHIDORI

        return GestureType.NONE
