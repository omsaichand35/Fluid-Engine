import threading
import cv2
import time
import sys
from pathlib import Path

# Ensure local app package is on the path (so core/engine modules resolve)
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT / "app"))

from app.window import WindowApp
from cv.camera import Camera
from cv.hand_tracker import HandTracker
from cv.gesture_recognizer import GestureRecognizer, GestureSequenceRecognizer
from cv.gesture_types import Gesture
from controller.jutsu_controller import JutsuController

def cv_loop(app):
    camera = Camera()
    tracker = HandTracker()
    recognizer = GestureRecognizer()
    seq_recognizer = GestureSequenceRecognizer(timeout=3.0)

    # Wait for the engine to be initialized by WindowApp
    while not hasattr(app, 'engine'):
        time.sleep(0.1)
        
    controller = JutsuController(app.engine)

    print("CV Thread Started.")

    while app.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        # Process hand tracking using direct solutions.hands API
        results = tracker.detect(frame)
        
        # Recognize gesture
        gesture = recognizer.recognize(results)
        seq_recognizer.add_gesture(gesture)
        sequence = seq_recognizer.get_sequence()
        
        # Draw for debug window
        display_frame = frame.copy()
        display_frame = tracker.draw_landmarks(display_frame, results)
        
        if results and results.multi_hand_landmarks and gesture != Gesture.NONE and gesture != Gesture.UNKNOWN:
            first_hand = results.multi_hand_landmarks[0]
            wrist = first_hand.landmark[0]
            wrist_x = int(wrist.x * display_frame.shape[1])
            wrist_y = int(wrist.y * display_frame.shape[0])
            
            # Draw prominent label right near the user's hand!
            cv2.putText(display_frame, str(gesture) + "!", (wrist_x - 50, wrist_y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # Draw Combo Sequence Output
        seq_text = "Sequence: " + " -> ".join([str(g) for g in sequence])
        cv2.putText(display_frame, seq_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
        top_text = f"Current Sign: {gesture}" if gesture != Gesture.NONE else "Scanning for signs..."
        cv2.putText(display_frame, top_text, (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Ninja Vision", display_frame)
        
        # We need a small waitKey to process OpenCV GUI events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            app.running = False
            break

        # Execute jutsu
        if controller.process_sequence(sequence):
            # Clear sequence after a successful jutsu combo execution
            seq_recognizer.clear()

    camera.release()
    cv2.destroyAllWindows()
    print("CV Thread Stopped.")

if __name__ == "__main__":
    app = WindowApp()
    
    # Start the computer vision pipeline in a background thread
    cv_thread = threading.Thread(target=cv_loop, args=(app,), daemon=True)
    cv_thread.start()
    
    # Run the pygame main loop
    app.run()
