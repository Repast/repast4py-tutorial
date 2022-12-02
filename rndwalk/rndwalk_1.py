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

    # def walk(self):
    #     if self.id == 10:
    #         print(f'WALKER: {self.uid} walking')


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
        # self.runner = schedule.init_schedule_runner(comm)
        # self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_stop(params['stop.at'])

        self.rank = comm.Get_rank()
        for i in range(params['walker.count']):
            # create and add the walker to the context
            walker = Walker(i, self.rank)
            self.context.add(walker)

        print(f'RANK: {self.rank}, SIZE: {self.context.size()[-1]}')
    
    # def step(self):
    #     for walker in self.context.agents():
    #         walker.walk()

    # def start(self):
    #     self.runner.execute()

def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
