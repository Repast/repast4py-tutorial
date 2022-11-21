from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt


class Walker(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)

    # 3: def __init__(self, local_id: int, rank: int, pt: dpt):
    # 3:    super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
    # 3:    self.pt = pt

    # 2: def walk(self):
    # 2:    if self.id < 5:
    # 2:        print(f'WALKER: {self.uid} walking')

    # 3: def walk(self, grid):
         # choose two elements from the OFFSET array
         # to select the direction to walk in the
         # x and y dimensions
    # 3:   xy_dirs = random.default_rng.choice(Walker.OFFSETS, size=2)
    # 3:   self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))
    # 3:   if self.id == 999:
    # 3:       print(f'{self.uid} walking at {self.pt}')


class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.context = ctx.SharedContext(comm)
        # 2: self.runner = schedule.init_schedule_runner(comm)
        # 2: self.runner.schedule_repeating_event(1, 1, self.step)
        # 2: self.runner.schedule_stop(params['stop.at'])

        # create a bounding box equal to the size of the entire global world grid
        # 3: box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        # 3: self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
        # 3:                                  occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        # 3: self.context.add_projection(self.grid)

        # 3: rng = repast4py.random.default_rng
        rank = comm.Get_rank()
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            # 3: pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, rank) # 3: , pt)
            self.context.add(walker)
            # 3: self.grid.move(walker, pt)

    # 2: def step(self):
    # 2:    for walker in self.context.agents():
    # 2:        walker.walk()
    # 3:        walker.walk(self.grid)

    # 2: def start(self):
    # 2:    self.runner.execute()

def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    # 2: model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
