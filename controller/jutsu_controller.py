import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv.gesture_types import GestureType

class JutsuController:
    def __init__(self, engine):
        self.engine = engine
        self.current_jutsu = GestureType.NONE

    def process_gesture(self, gesture: GestureType):
        # Prevent repeatedly calling the same jutsu while holding the pose
        if gesture == self.current_jutsu and gesture != GestureType.WIND:
            # We allow WIND to be re-triggered repeatedly as it's a motion gesture
            return
            
        self.current_jutsu = gesture
        
        if gesture == GestureType.RASENGAN:
            if hasattr(self.engine, 'spawn_rasengan'):
                self.engine.spawn_rasengan()
            else:
                print("Jutsu Triggered: RASENGAN! (waiting for engine effect)")
                
        elif gesture == GestureType.CHIDORI:
            if hasattr(self.engine, 'spawn_chidori'):
                self.engine.spawn_chidori()
            else:
                print("Jutsu Triggered: CHIDORI! (waiting for engine effect)")
                
        elif gesture == GestureType.FIREBALL:
            if hasattr(self.engine, 'spawn_fireball'):
                self.engine.spawn_fireball()
            else:
                print("Jutsu Triggered: FIREBALL! (waiting for engine effect)")
                
        elif gesture == GestureType.WIND:
            if hasattr(self.engine, 'spawn_wind'):
                self.engine.spawn_wind()
            else:
                print("Jutsu Triggered: WIND TORNADO! (waiting for engine effect)")
                
        elif gesture == GestureType.AURA:
            if hasattr(self.engine, 'spawn_aura'):
                self.engine.spawn_aura()
            else:
                print("Jutsu Triggered: CHAKRA AURA! (waiting for engine effect)")
