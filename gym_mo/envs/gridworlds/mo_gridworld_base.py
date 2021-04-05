from typing import Tuple
from gym_mo.envs.gridworlds import gridworld_base
from gym_mo.envs.gridworlds.gridworld_base import Gridworld, Viewport

import numpy as np
import time
import gin

class MOEnvDummy():

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        obs = np.zeros(shape=self.observation_space.shape)
        return obs

@gin.configurable()
class MOGridworld(Gridworld):
    """Base class for multi objective gridworld environments."""

    def __init__(self,
                 map,
                 object_mapping,
                 viewport=Viewport(),
                 from_pixels: bool = True,
                 inflation=1,
                 random_items=[],
                 random_items_frame=0,
                 init_agents=[],
                 agent_start: gridworld_base.Position = [0, 0],
                 agent_color: gridworld_base.Color = (0.0, 0.0, 255.0),
                 encounter_other_agents=False,
                 max_steps: int = 50):
        super(MOGridworld, self).__init__(map=map,
                                          object_mapping=object_mapping,
                                          viewport=viewport,
                                          from_pixels=from_pixels,
                                          inflation=inflation,
                                          random_items=random_items,
                                          random_items_frame=random_items_frame,
                                          init_agents=init_agents,
                                          agent_start=agent_start,
                                          agent_color=agent_color,
                                          encounter_other_agents=encounter_other_agents,
                                          max_steps=max_steps)
        self.agents = init_agents

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, float]:

        self.step_count += 1

        new_pos = self.get_new_pos_from_action(action, self.agent_pos)

        reward = np.zeros(shape=(self.max_idx + 1,), dtype='uint8')
        reward[0] = 1 # Added one time step
        if self.is_walkable(new_pos[0], new_pos[1]):
            self.agent_pos = new_pos
        else:
            reward[1] = 1 # Agent tried to walk in illegal direction

        # Update grid agents
        agent_rewards = 0
        for agent in self.agents:
            agent.step(self)
            agent_rewards += agent.reward
            reward[agent.idx] += agent.reward
            agent.reset_reward()

        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()

        idxs = self.encounter_object_idx(self.agent_pos[0], self.agent_pos[1], self.encounter_other_agents)
        for idx in idxs:
             reward[idx] += 1

        return (obs, reward, agent_rewards)

    def encounter_object_idx(self, column, row, encounter_agents=True):
        idxs = []
        if self.within_bounds(column, row) and self.grid[row, column] is not None:
            idxs.append(self.grid[row, column].idx)
            if self.grid[row, column].is_consumed:
                self.grid[row, column] = None

        if encounter_agents:
            for agent in self.agents:
                if (agent.position[0]==column) and (agent.position[1]==row):
                    idxs.append(agent.idx)
                    if agent.is_consumed:
                        self.agents.remove(agent)

        return idxs

    def reset(self) -> np.ndarray:
        self.load_map(self.map, self.object_mapping)
        self.agent_pos = self.agent_start
        for agent in self.agents:
            agent.reset()
        self.step_count = 0
        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()
        return obs



if __name__=="__main__":
    my_grid = MOGridworld(gridworld_base.TEST_MAP, 
                          gridworld_base.TEST_MAPPING)

    done = False
    my_grid.reset()
    while not done:
        _, r, done, _ = my_grid.step(my_grid.action_space.sample())
        my_grid.render()
        time.sleep(0.5)
    
