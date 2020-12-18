import time

from gym_mo.envs.gridworlds.mo_gridworld_base import MOGridworld
from gym_mo.envs.gridworlds.gridworld_base import GridObject, HunterAgent, Position

import numpy as np

GATHERING_MAPPING = {
    '#': GridObject(True, False, 0, (255.0, 255.0, 255.0), 1),
    'o': GridObject(True, True, 0, (0.0, 255.0, 0.0), 2),
    'p': GridObject(True, True, 1, (255.0, 0.0, 0.0), 3),
    'q': GridObject(True, True, 0, (255.0, 255.0, 0.0), 4),
    ' ': None
}

GATHERING_MAP = [
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
]


class MOGatheringEnv(MOGridworld):

    def __init__(self,
                 from_pixels: bool = True,
                 agent_start: Position = [0,0],
                 agent_color: tuple = (0.0, 0.0, 255.0),
                 preference: np.ndarray = np.array([-1,-5,+20,-20,-20,+0]),
                 random_items: list = ['p','o','p','o','p','o','q','q'],
                 random_items_frame: int = 2,
                 agents=[]):

        agent0 = HunterAgent(3, True, False, 0, (255.0, 0.0, 255.0), 5)
        agent0.set_position([7,7])

        GATHERING_AGENTS = [agent0]

        super(MOGatheringEnv, self).__init__(map=GATHERING_MAP,
                                             object_mapping=GATHERING_MAPPING,
                                             random_items=random_items,
                                             random_items_frame=random_items_frame,
                                             from_pixels=from_pixels,
                                             init_agents=GATHERING_AGENTS,
                                             agent_start=agent_start,
                                             agent_color=agent_color,
                                             preference=preference,
                                             max_steps=30, include_agents=False)


if __name__=="__main__":
    my_grid = MOGatheringEnv(from_pixels=True)

    done = False
    my_grid.reset()
    while not done:
        obs, r, done, _ = my_grid.step(my_grid.action_space.sample())
        my_grid.render()
        time.sleep(0.5)
