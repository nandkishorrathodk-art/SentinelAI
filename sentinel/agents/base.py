import asyncio
from typing import Any, Dict, List, Optional
import abc

class CognitiveSubAgent(abc.ABC):
    """
    Base class for SentinelAI harvester agents.
    Provides async execution, message passing, observation intake, and autonomous decision loops.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory: List[Dict[str, Any]] = []
        self.is_running = False

    async def send_message(self, recipient_id: str, message: Any) -> None:
        """Sends a message to another agent or subsystem."""
        # Implementation to route messages via the system's message bus
        pass

    async def receive_observation(self, observation: Any) -> None:
        """Intakes an observation from the environment or another agent."""
        self.memory.append({"role": "observation", "content": observation})

    @abc.abstractmethod
    async def step(self) -> None:
        """Executes a single step of the autonomous decision loop."""
        pass

    async def run_loop(self) -> None:
        """Runs the continuous autonomous loop."""
        self.is_running = True
        while self.is_running:
            await self.step()
            await asyncio.sleep(0.1)  # Prevent tight looping and yield execution

    def stop(self) -> None:
        """Stops the autonomous loop."""
        self.is_running = False
