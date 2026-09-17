"""
SentinelLoop manager.
Orchestrates AWAKE, PATROL, and DREAM states.
"""
from enum import Enum
import threading
import time
from .patrol import PatrolEngine
from .dream import DreamWorker

class SystemState(Enum):
    AWAKE = "AWAKE"
    PATROL = "PATROL"
    DREAM = "DREAM"

class SentinelLoop:
    """
    Main loop manager for SentinelAI state transitions.
    """
    def __init__(self, patrol_engine: PatrolEngine, dream_worker: DreamWorker):
        self.state = SystemState.AWAKE
        self.patrol_engine = patrol_engine
        self.dream_worker = dream_worker
        self._running = False
        self._lock = threading.Lock()

    def set_state(self, new_state: SystemState) -> None:
        """Transitions the system to a new state."""
        with self._lock:
            self.state = new_state
            print(f"SentinelLoop transitioned to {self.state.name}")

    def run(self) -> None:
        """Starts the sentinel loop."""
        self._running = True
        while self._running:
            with self._lock:
                current_state = self.state
            
            if current_state == SystemState.AWAKE:
                time.sleep(1)
            elif current_state == SystemState.PATROL:
                self.patrol_engine.step()
                time.sleep(5)
            elif current_state == SystemState.DREAM:
                self.dream_worker.process()
                self.set_state(SystemState.PATROL)

    def stop(self) -> None:
        """Stops the loop."""
        self._running = False
