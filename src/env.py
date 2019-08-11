#!/usr/bin/env python

import numpy as np
from gym import spaces

from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space

from .env_wrappers import AnimalAIWrapper, TransposeImage


BIG_NUM = np.iinfo(np.int32).max


class AnimalEnvWrapper(EnvironmentSpec):
    def __init__(
        self,
        history_len=1,
        frame_skip=1,
        reward_scale=1,
        action_mean=None,
        action_std=None,
        visualize=False,
        mode="train",
        sampler_id=None,
        **params
    ):
        super().__init__(visualize=visualize, mode=mode, sampler_id=sampler_id)

        if not self._visualize:
            # # virtual display hack
            from pyvirtualdisplay import Display
            from pyvirtualdisplay.randomize import Randomizer
            self.display = Display(randomizer=Randomizer())
            self.display.start()

        env = AnimalAIWrapper(worker_id=self._sampler_id, **params)
        env = TransposeImage(env, op=[2, 0, 1])
        self.env = env

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._visualize = visualize
        self._reward_scale = reward_scale

        self.action_mean = np.array(action_mean) \
            if action_mean is not None else None
        self.action_std = np.array(action_std) \
            if action_std is not None else None

        self._prepare_spaces()

    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> spaces.space.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.space.Space:
        return self._state_space

    @property
    def action_space(self) -> spaces.space.Space:
        return self._action_space

    def _prepare_spaces(self):
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

        self._state_space = extend_space(
            self._observation_space, self._history_len
        )

    def _process_action(self, action):
        if self.action_mean is not None \
                and self.action_std is not None:
            action = action * (self.action_std + 1e-8) + self.action_mean
        return action

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        reward = 0
        action = self._process_action(action)
        for i in range(self._frame_skip):
            observation, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        info["raw_reward"] = reward
        reward *= self._reward_scale
        return observation, reward, done, info
