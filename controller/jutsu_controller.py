import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv.gesture_types import Gesture

class JutsuController:
    def __init__(self, engine):
        self.engine = engine
        self.current_jutsu = Gesture.NONE

    def process_sequence(self, sequence):
        if not sequence:
            return False

        # Combo logic
        if len(sequence) >= 2:
            combo = sequence[-2:]
            
            # Fireball
            if combo == [Gesture.SNAKE, Gesture.TIGER]:
                if hasattr(self.engine, 'spawn_fireball'):
                    self.engine.spawn_fireball()
                else:
                    print("Jutsu Combo Triggered: FIREBALL!")
                return True
                
            # Random combo for Chidori (Dog -> Ram)
            if combo == [Gesture.DOG, Gesture.RAM]:
                if hasattr(self.engine, 'spawn_chidori'):
                    self.engine.spawn_chidori()
                else:
                    print("Jutsu Combo Triggered: CHIDORI!")
                return True

        # Process single signs logic for backwards compatibility / simple actions
        latest = sequence[-1]
        
        if latest == self.current_jutsu and latest != Gesture.WIND:
            return False
            
        self.current_jutsu = latest
        
        if latest == Gesture.RASENGAN:
            if hasattr(self.engine, 'spawn_rasengan'):
                self.engine.spawn_rasengan()
            else:
                print("Jutsu Triggered: RASENGAN!")
                
        elif latest == Gesture.CHIDORI:
            if hasattr(self.engine, 'spawn_chidori'):
                self.engine.spawn_chidori()
            else:
                print("Jutsu Triggered: CHIDORI!")
                
        elif latest == Gesture.FIREBALL:
            if hasattr(self.engine, 'spawn_fireball'):
                self.engine.spawn_fireball()
            else:
                print("Jutsu Triggered: FIREBALL!")
                
        elif latest == Gesture.WIND:
            if hasattr(self.engine, 'spawn_wind'):
                self.engine.spawn_wind()
            else:
                print("Jutsu Triggered: WIND TORNADO!")
                
        elif latest == Gesture.AURA:
            if hasattr(self.engine, 'spawn_aura'):
                self.engine.spawn_aura()
            else:
                print("Jutsu Triggered: CHAKRA AURA!")
                
        return True
