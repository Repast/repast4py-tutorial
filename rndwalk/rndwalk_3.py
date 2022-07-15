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

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
        self.pt = pt

    def walk(self):
        print(f'{self.uid} walking on {self.pt}')
        

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

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)

        rank = comm.Get_rank()
        rng = repast4py.random.default_rng
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, rank, pt)
            self.context.add(walker)
            self.grid.move(walker, pt)

    def step(self):
        for walker in self.context.agents():
            walker.walk()

    def start(self):
        self.runner.execute()
    

def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
