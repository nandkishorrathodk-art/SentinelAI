"""
Offline consolidation worker.
"""
from sentinel.memory.consolidator import DreamConsolidator

class DreamWorker:
    """
    Executes deep consolidation tasks during DREAM state.
    """
    def __init__(self, consolidator: DreamConsolidator):
        self.consolidator = consolidator

    def process(self) -> None:
        """Runs the consolidation process."""
        print("DreamWorker: Starting deep consolidation...")
        self.consolidator.run_consolidation()
        print("DreamWorker: Consolidation complete.")
