from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt


@dataclass
class ColocationLog:
    total_colocs: int = 0
    min_colocs: int = 0
    max_colocs: int = 0


class Walker(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
        self.pt = pt

    def walk(self, grid):
        # choose two elements from the OFFSET array
        # to select the direction to walk in the
        # x and y dimensions
        xy_dirs = random.default_rng.choice(Walker.OFFSETS, size=2)
        self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))
        if self.local_rank != self.uid_rank:
            print(f'{self.uid} walking at {self.pt} on rank {self.local_rank}')

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.

        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.pt.coordinates)

    def count_colocations(self, grid, coloc_log: ColocationLog):
        # subtract self
        num_here = grid.get_num_agents(self.pt) - 1
        coloc_log.total_colocs += num_here
        if num_here < coloc_log.min_colocs:
            coloc_log.min_colocs = num_here
        if num_here > coloc_log.max_colocs:
            coloc_log.max_colocs = num_here
        if self.id == 999:
            print(f'{self.uid} colocated with {num_here} walkers')



walker_cache = {}

def restore_walker(walker_data: Tuple):
    """
    Args:
        walker_data: tuple containing the data returned by Walker.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = walker_data[0]
    pt_array = walker_data[1]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid in walker_cache:
        walker = walker_cache[uid]
    else:
        walker = Walker(uid[0], uid[2], pt)

    walker.pt = pt
    return walker

        

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

        self.coloc_log = ColocationLog()
        loggers = logging.create_loggers(self.coloc_log, op=MPI.SUM, names={'total_colocs': 'total'}, rank=rank)
        loggers += logging.create_loggers(self.coloc_log, op=MPI.MIN, names={'min_colocs': 'min'}, rank=rank)
        loggers += logging.create_loggers(self.coloc_log, op=MPI.MAX, names={'max_colocs': 'max'}, rank=rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['coloc_log_file'])

        # count the initial colocations at time 0 and log
        for walker in self.context.agents():
            walker.count_colocations(self.grid, self.coloc_log)
        self.data_set.log(0)
        self.coloc_log.max_colocs = self.coloc_log.min_colocs = self.coloc_log.total_colocs = 0
        self.runner.schedule_end_event(self.data_set.close)

    def step(self):
        for walker in self.context.agents():
            walker.walk(self.grid)

        self.context.synchronize(restore_walker)

        for walker in self.context.agents():
            walker.count_colocations(self.grid, self.coloc_log)

        tick = self.runner.schedule.tick
        self.data_set.log(tick)
        # clear the log counts for the next tick
        self.coloc_log.max_colocs = self.coloc_log.min_colocs = self.coloc_log.total_colocs = 0


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
