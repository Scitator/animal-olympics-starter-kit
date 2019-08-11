from catalyst.rl import registry

from src.env import AnimalEnvWrapper
from src.actor import AnimalActor
from src.critic import AnimalStateCritic

registry.Environment(AnimalEnvWrapper)
registry.Agent(AnimalActor)
registry.Agent(AnimalStateCritic)
