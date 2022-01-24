# Copyright 2019 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import collections
import random
import sys

import numpy as np

import gym
import math
import numpy
from gym import spaces
from overrides import overrides

from .helpers import Instance, inverse_odds

INSTANCE_COSTS_PER_HOUR = {
    'c3.large': 0.192,
}
SCALING_STEP_SIZE_IN_SECONDS = 300  # Minimum resolution in AWS CloudWatch is 5 minutes

INPUTS = {
    'PRODUCTION_DATA': {
        'function': lambda: None,
        'options': {
            'path': 'data/worker_one.xlsx',
            'sheet': 'Input',
            'column': 'MessagesReceived'
        }
    },
    'SINE_CURVE': {
        'function': lambda step, max_influx, offset: math.ceil((numpy.sin(float(step + offset) * 0.01) + 1) * max_influx / 2),
        'options': {
        },
    },
    'RANDOM': {
        'function': lambda step, max_influx, offset: random.randint(offset, max_influx),
        'options': {
        },
    }
}


class ContinuousScalingEnv(gym.Env):
    DEFAULTS = {
        'max_instances': 100.0,
        'min_instances': 2.0,
        'capacity_per_instance': 87,
        'cost_per_instance_per_hour': INSTANCE_COSTS_PER_HOUR['c3.large'],
        'step_size_in_seconds': SCALING_STEP_SIZE_IN_SECONDS,
        'discrete_actions': (-1, 0, 1),
        'input': INPUTS['SINE_CURVE'],
        'offset': 500,
        'size': (300, 250),
        'change_rate': 1000,
        'max_steps' : 20000
    }

    @overrides
    def __init__(self, *args, **kwargs):
        self.sim_size = (300, 250)
        # Set options and defaults
        self.scaling_env_options = {
            **self.DEFAULTS,
            **(kwargs.pop('scaling_env_options', {})),
        }
        # self.actions = self.scaling_env_options['discrete_actions']
        # self.num_actions = len(self.actions)
        
        self.action_low = self.scaling_env_options['min_instances'] / self.scaling_env_options['max_instances']
        self.action_high = 1.0
        
        self.action_space = spaces.Box(low=self.action_low, high=1.0, shape=(1,)) #spaces.Discrete(self.num_actions)
        self.observation_size = 5
        self.observation_space = spaces.Box(low=0.0, high=sys.float_info.max, shape=(5,))
        self.window = None
        
        # self.noise = OUNoise(self.action_space)
        self.action = 0.0
        self.action_noised = 0.0

        self.max_instances = self.scaling_env_options['max_instances']
        self.min_instances = self.scaling_env_options['min_instances']
        self.capacity_per_instance = self.scaling_env_options["capacity_per_instance"]

        self.offset = self.scaling_env_options['offset']
        self.sim_size = self.scaling_env_options['size']
        self.change_rate = self.scaling_env_options['change_rate']
        self.influx_range = ((self.max_instances / 2) * self.capacity_per_instance) - self.offset
        self.max_influx = self.offset + self.influx_range
        self.max_history = math.ceil(self.sim_size[0])
        
        self.cost_per_instance_per_step = self.scaling_env_options["cost_per_instance_per_hour"] / 12
        self.max_steps = self.scaling_env_options['max_steps']
        
        self.process_reward = 0
        self.cost_reward = 0
        self.processed_items = 0

        super().__init__(*args, **kwargs)

        self.reset()

    @overrides
    def step(self, action):
        self.step_idx += 1
        
        if self.step_idx % self.change_rate == 0:
            self.influx = self.__next_influx()

        self.hi_influx.append(self.influx)
        self.hi_instances.append(self.instances)
        total_items = self.influx + self.queue_size

        self.total_capacity = self.instances * self.capacity_per_instance
        self.processed_items = min(total_items, self.total_capacity)

        self.hi_load.append(self.load)
        self.load = math.ceil(float(self.processed_items) / float(self.total_capacity) * 100)

        self.hi_queue_size.append(self.queue_size)
        self.queue_size = total_items - self.processed_items

        self.total_cost += self.instances * self.cost_per_instance_per_step

        # ??? the action is based on previous observation, it can't be used in this step, can't be used in next step, so actually 2 steps delay are applied here,
        # in simplified model, maximum 1 step delay, other wise, will involve complex transitional problem, e.g. shutdown an instance during its start process
        # ??? observation after do_action, but new instance num didn't do anything, which can't reflect real situation of current timestep 
        
        observation = self.__get_observation()
        reward = self.__get_reward()
        self.__do_action(action)

        done = int(self.queue_size > self.max_influx * 50 or self.step_idx >= self.max_steps)

        return observation, reward, done, {}

    @overrides
    def reset(self):
        # self.instances = []
        # for _ in range(int(self.scaling_env_options['max_instances'] / 2)):
        #     self.instances.append(
        #         Instance(
        #             step=0,
        #             cost_per_hour=self.capacity_per_instance
        #         )
        #     )
        
        # self.noise.reset()
        
        self.hi_instances = collections.deque(maxlen=self.max_history)
        # self.scaling_actions = collections.deque(maxlen=self.max_history)
        # self.scaling_actions.appendleft(0)
        self.instances = random.randint(self.min_instances, self.max_instances)
        self.total_capacity = self.instances * self.capacity_per_instance
        self.load = 0.0
        self.influx_derivative = 0.0
        self.queue_size = 0.0
        self.error = 0.0
        self.step_idx = 0
        self.hi_queue_size = collections.deque(maxlen=self.max_history)
        self.hi_influx = collections.deque(maxlen=self.max_history)
        self.hi_load = collections.deque(maxlen=self.max_history)
        self.influx = self.__next_influx()
        self.reward = 0.0
        self.total_cost = 0.0
        self.collected_rewards = collections.deque(maxlen=self.max_history * 10)

        return self.__get_observation()

    @overrides
    def render(self, mode='human'):
        if len(self.collected_rewards) == 0:
            # skip rendering without at least one step
            return

        from environments.scaling.rendering import PygletWindow

        stats = self.get_stats()
        if self.window is None:
            self.window = PygletWindow(self.sim_size[0] + 20, self.sim_size[1] + 20 + 20 * len(stats))

        self.window.reset()

        x_offset = 10
        sim_height = self.sim_size[1]

        self.window.rectangle(x_offset, 10, self.sim_size[0], sim_height)

        max_influx_axis = max(max(self.hi_influx), max(self.hi_queue_size))
        max_instance_axis = max(self.hi_instances)

        self.window.text(str(max_influx_axis), 1, 1, font_size=5)
        self.window.text(str(max_instance_axis), self.sim_size[0] + 5, 1, font_size=5)

        influx_scale_factor = float(sim_height) / float(max_influx_axis)
        instance_scale_factor = float(sim_height) / float(max_instance_axis)

        self.draw_data(influx_scale_factor, instance_scale_factor, x_offset)

        stats_offset = sim_height + 15
        for txt in stats:
            self.window.text(txt, x_offset, stats_offset, font_size=8)
            stats_offset += 20

        self.window.update()

    def draw_data(self, influx_scale_factor, instance_scale_factor, x_offset):
        from environments.scaling.rendering import RED, BLACK, GREEN

        prev_queue_size = 0
        prev_influx = 0
        prev_instances = 0
        y_offset = self.sim_size[1] + 5
        for influx, instances, queue_size in zip(self.hi_influx, self.hi_instances, self.hi_queue_size):
            x_offset += 1

            qs_lp = (x_offset - 1, (y_offset - 2) - math.ceil(influx_scale_factor * float(prev_queue_size)))
            qs_rp = (x_offset, (y_offset - 2) - math.ceil(influx_scale_factor * float(queue_size)))
            self.window.line(qs_lp, qs_rp, color=RED)

            i_lp = (x_offset - 1, (y_offset - 1) - math.ceil(influx_scale_factor * float(prev_influx)))
            i_rp = (x_offset, (y_offset - 1) - math.ceil(influx_scale_factor * float(influx)))
            self.window.line(i_lp, i_rp, color=BLACK)

            s_lp = (x_offset - 1, (y_offset - 1) - instance_scale_factor * prev_instances)
            s_rp = (x_offset, (y_offset - 1) - instance_scale_factor * instances)
            self.window.line(s_lp, s_rp, color=GREEN)

            prev_queue_size = queue_size
            prev_influx = influx
            prev_instances = instances

    def get_stats(self):
        # actions = "a: " + ' '.join(str(a) for a in self.scaling_actions)
        return [
            "frame             = %d" % self.step_idx,
            "avg reward        = %.5f" % (sum(self.collected_rewards) / len(self.collected_rewards)),
            "instance cost     = %d $" % math.ceil(self.total_cost),
            "load              = %d" % self.load,
            "action            = %.3f" % self.action,
            "action_noised     = %.3f" % self.action_noised,
            "instances         = %d" % self.instances,
            "reward            = %d" % self.reward,
            "influx            = %d" % self.influx,
            "influx_derivative = %.2f" % self.influx_derivative,
            # "actions q         = %s" % actions,
            "avg queue size    = %.3f" % (sum(self.hi_queue_size, ) / len(self.hi_queue_size)),
            "avg instances     = %.3f" % (sum(self.hi_instances) / len(self.hi_instances)),
            "avg load          = %.3f" % (sum(self.hi_load) / len(self.hi_load)),
            "process_reward      = %.3f" % self.process_reward,
            "cost_reward       = %.3f" % self.cost_reward
        ]

    @overrides
    def close(self):
        if self.window:
            self.window.close()
            self.window = None

    def __next_influx(self):
        return 2000
        return self.scaling_env_options['input']['function'](self.step_idx, self.max_influx, self.offset)

    def __do_action(self, action):
        
        self.action = action[0]
        # print("@@@@@@ step {}: action: {}".format(self.step_idx,action))
        # self.action_noised = self.noise.get_action(action)
        
        # assert self.action_low <= action <= self.action_high
        self.action = np.clip(self.action, self.action_low, self.action_high)
        self.instances = int(self.action * self.max_instances)

    def __get_observation(self):
        observation = numpy.zeros(self.observation_size)
        observation[0] = self.instances / self.max_instances
        observation[1] = self.load / 100
        observation[2] = self.total_capacity
        observation[3] = self.influx
        observation[4] = self.queue_size
        
        # observation[0] = self.instances * self.capacity_per_instance
        # observation[1] = self.max_instances * self.capacity_per_instance
        # observation[2] = self.influx
        # observation[3] = self.queue_size
        return observation

    def __get_reward(self):
        normalized_load = self.load / 100
        # num_instances_normalized = self.instances / self.max_instances
        # total_reward = (-1 * (1 - normalized_load)) * num_instances_normalized
        # total_reward -= inverse_odds(self.queue_size)
        
        self.process_reward = self.processed_items - self.influx
        self.cost_reward = -(1 - normalized_load) * self.instances * self.capacity_per_instance
        
        total_reward = self.process_reward + 0.3 * self.cost_reward
        # total_reward = self.process_reward
        self.collected_rewards.append(total_reward)
        
        self.reward = total_reward
        return total_reward
