import itertools
import numpy as np
import gym

from animalai.envs import UnityEnvironment, ArenaConfig


class AnimalAIWrapper(gym.Env):
    def __init__(
        self,
        worker_id,
        env_path,
        config_path,
        reduced_actions=False,
        docker_training=False,
    ):
        super(AnimalAIWrapper, self).__init__()
        self.config = ArenaConfig(config_path)
        self.time_limit = self.config.arenas[0].t

        self.env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=worker_id,
            n_arenas=1,
            arenas_configurations=self.config,
            docker_training=docker_training,
        )

        lookup_func = lambda a: {"Learner": np.array([a], dtype=float)}
        if reduced_actions:
            lookup = itertools.product([0, 1], [0, 1, 2])
        else:
            lookup = itertools.product([0, 1, 2], repeat=2)
        lookup = dict(enumerate(map(lookup_func, lookup)))
        self.action_map = lambda a: lookup[a]

        self.observation_space = gym.spaces.Box(
            0, 255,
            [84, 84, 3],
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(len(lookup))
        self.t = 0

    def _process_state(self, state):
        img = 255 * state["Learner"].visual_observations[0][0]
        vec = state["Learner"].vector_observations[0]
        r = state["Learner"].rewards[0]
        done = state["Learner"].local_done[0]
        return np.uint8(img), vec, r, done

    def reset(self):
        self.t = 0
        img, vec, r, done = self._process_state(
            self.env.reset(arenas_configurations=self.config))
        while done:
            img, vec, r, done = self._process_state(
                self.env.reset(arenas_configurations=self.config))
        return img

    def step(self, action):
        obs, vec, r, done = self._process_state(
            self.env.step(vector_action=self.action_map(action.item())))
        self.t += 1
        done = done or self.t >= self.time_limit
        return obs, r, done, {}


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super().__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super().__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])
