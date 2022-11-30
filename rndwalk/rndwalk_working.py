from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt


# 5: @dataclass
# 5: class DistanceLog:
# 5:    min_distance: float = 0
# 5:    max_distance: float = 0


class Walker(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)

    # 3: def __init__(self, local_id: int, rank: int, pt: dpt):
    # 3:    super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
    # 3:    self.pt = pt
    # 5:    self.starting_pt = pt

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

    # 4:   if self.local_rank != self.uid_rank:
    # 4:       print(f'{self.uid} walking at {self.pt} on rank {self.local_rank}')

    # 4: def save(self) -> Tuple:
    # 4:    """Saves the state of this Walker as a Tuple.

    # 4:    Returns:
    # 4:        The saved state of this Walker.
    # 4:    """
    # 4:    return (self.uid, self.pt.coordinates)

    # 5: def distance(self):
    # 5:    return np.linalg.norm(self.starting_pt.coordinates - self.pt.coordinates)



# 4: walker_cache = {}

# 4: def restore_walker(walker_data: Tuple):
# 4:    """
# 4:    Args:
# 4:        walker_data: tuple containing the data returned by Walker.save.
# 4:    """
# 4:    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
# 4:    uid = walker_data[0]
# 4:    pt_array = walker_data[1]
# 4:    pt = dpt(pt_array[0], pt_array[1], 0)
# 4:
# 4:    if uid in walker_cache:
# 4:        walker = walker_cache[uid]
# 4:    else:
# 4:        walker = Walker(uid[0], uid[2], pt)
# 4:
# 4:    walker.pt = pt
# 4:    return walker



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
        self.rank = comm.Get_rank()
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            # 3: pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, self.rank)
            # 3: walker = Walker(i, self.rank, pt)
            self.context.add(walker)
            # 3: self.grid.move(walker, pt)

        # 5: self.log = DistanceLog()
        # 5: loggers = logging.create_loggers(self.log, op=MPI.MIN, names={'min_distance': None}, rank=self.rank)
        # 5: loggers += logging.create_loggers(self.log, op=MPI.MAX, names={'max_distance': None}, rank=self.rank)
        # 5: self.data_set = logging.ReducingDataSet(loggers, comm, params['log.file'])
        # 5: self.runner.schedule_end_event(self.data_set.close)

    # 2: def step(self):
    # 2:    for walker in self.context.agents()
    # 2:        walker.walk()
    
    # 3: def step(self):

    # 5:    self.log.max_distance = float('-inf')
    # 5:    self.log.min_distance = float('inf')

    # 3:    for walker in self.context.agents()
    # 3:        walker.walk(self.grid)
    # 5:        self.log_distance(walker)
    
    # 4:    self.context.synchronize(restore_walker)

    # 5:    tick = self.runner.schedule.tick
    # 5:    self.data_set.log(tick)

    # 2: def start(self):
    # 2:    self.runner.execute()

    # 5: def log_distance(self, walker):
    # 5:     distance = walker.distance()
    # 5:     if distance < self.log.min_distance:
    # 5:         self.log.min_distance = distance
    # 5:     if distance > self.log.max_distance:
    # 5:         self.log.max_distance = distance

def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    # 2: model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
