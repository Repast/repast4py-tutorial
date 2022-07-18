# Tutorial Steps

The tutorial builds a version of the Random Walk demonstration model.
The simulation itself consists of a number of agents moving at random around a two-dimensional grid 
and logging the aggregate and agent-level colocation counts. Each timestep the following occurs:

1. All the agents (walkers) choose a random direction and move one unit in that direction.

2. All the agents count the number of other agents they meet at their current
location by determining the number of colocated agents at their grid locations.

3. The sum, minimum, and maxiumum number of co-located agents are calculated across
all process ranks, and these values are logged as the total, minimum, and maximum colocation
values.

The code consists of the following components:

1. A `Walker` class that implements the agent state and behavior.
2. A `Model` class responsible for initialization and managing the simulation.
3. A `restore_walker` function used to create an individual Walker when that Walker has moved (i.e., walked) to another process.
4. A `run` function that creates and starts the simulation.
5. An `if name == "main"` block that allows the simulation to be run from the command line.


The tutorial code begins with a skeleton, and we progressively add code to that
to implement the components. The code for the completed is specified at the beginning of each step.

## Step 0

Completed code in `rndwalk_0.py`.

1. Open a terminal in the binder Jupyter Lab launcher, and do
the following:

```bash
$ cd rndwalk
$ cp rndwalk_0.py rndwalk.py
$ python rndwalk.py random_walk.yaml

{'random.seed': 42, 'stop.at': 50, 'walker.count': 1000, 'world.width': 2000, 'world.height': 2000, 'coloc_log_file': 'output/coloc_log.csv'}
```

The skeleton parses the parameters from a yaml file and prints them out.

## Step 1

Completed code in `rndwalk_1.py`.

Step 1 begins the Walker agent implementation and
creates a population of Walker agents in the Model.

1. Add a minimal Walker to the code.

```python
class Walker(core.Agent):

    TYPE = 0

    def __init__(self, local_id: int, rank: int):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
```

2. In `Model.__init__()` create a context and the walkers

```python
self.context = ctx.SharedContext(comm)
rank = comm.Get_rank()

for i in range(params['walker.count']):
    # create and add the walker to the context
    walker = Walker(i, rank)
    self.context.add(walker)
    print(walker.uid)
```

```bash
$ python rndwalk.py random_walk.yaml
(0, 0, 0)
(1, 0, 0)
...
$ mpirun -n 2 python rndwalk.py random_walk.yaml
(0, 0, 0)
(1, 0, 0)
...
(0, 0, 1)
(1, 0, 1)
...
```

Notice how in the second case we have 1K agents on each process rank (0 and 1).

## Step 2

Completed code in `rndwalk_2.py`.

Step 2 continues the Walker implementation with a initial walk method,
and schedules that method to execute all the agents via the Model.

1. Add a walk method to Walker

```python
def walk(self):
    print(f'{self.uid} walking')
```

2. Add `Model.step()` to walk the Walkers.

```python
def step(self):
    for walker in self.context.agents():
        walker.walk()
```

3. In `Model.__init__()` create the schedule and schedule the step method

```python
self.runner = schedule.init_schedule_runner(comm)
self.runner.schedule_repeating_event(1, 1, self.step)
self.runner.schedule_stop(params['stop.at'])
```

4. Add a `start` method to `Model` to start the schedule

```python
def start(self):
    self.runner.execute()
```

4. In `def run()` call `Model.start()`

```python
def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()
```

## Step 3

Completed code in `rndwalk_3.py`.

Step 3 adds the 2D grid on which the Walkers walk.

1. In `Model.__init__()` below the schedule code, initialize the `SharedGrid`.

```python
# create a bounding box equal to the size of the entire global world grid
box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
# create a SharedGrid of 'box' size with sticky borders that allows multiple agents
# in each grid location.
self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
self.context.add_projection(self.grid)
```

2. Update the Walker creation code in `Model.__init__` to 
place the Walkers at a random location on the grid.

```python
rng = repast4py.random.default_rng
for i in range(params['walker.count']):
    # get a random x,y location in the grid
    pt = self.grid.get_random_local_pt(rng)
    # create and add the walker to the context
    walker = Walker(i, rank, pt)
    self.context.add(walker)
    self.grid.move(walker, pt)
```

3. Update the Walker's constructor to accept the grid point:

```python
def __init__(self, local_id: int, rank: int, pt: dpt):
    super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
    self.pt = pt
```

4. And update `Walker.walk` to display the point

```python
def walk(self):
    print(f'{self.uid} walking on {self.pt}')
```


## Step 4

Completed code in `rndwalk_4.py`.

Step 4 adds the Walker walking around the 2D grid.

1. In `Model.step()` pass `self.grid` to `Walker.walk`:

```python
def step(self):
    for walker in self.context.agents():
        walker.walk(self.grid)
```

2. Update `Walker.walk` with the `grid` argument and implement the
random movement.

```python
OFFSETS = np.array([-1, 1])

def walk(self, grid):
    # choose two elements from the OFFSET array
    # to select the direction to walk in the
    # x and y dimensions
    xy_dirs = random.default_rng.choice(Walker.OFFSETS, size=2)
    self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0)
    if self.id < 10:
        print(f'{self.uid} walking at {self.pt}')
```

## Step 5

Completed code in `rndwalk_5.py`.

Step 5 adds multiprocess synchronization so that Walkers can walk out of their
local area into that controlled by another process.

1. Add the `save` method to `Walker`.

```python
def save(self) -> Tuple:
    """Saves the state of this Walker as a Tuple.

    Returns:
        The saved state of this Walker.
    """
    return (self.uid, self.pt.coordinates)
```

2. Add a `restore_walker` function to create a `Walker` from the
data returned from `save`.

```python
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
```

3. Add synchronization to `Model.step` after the iteration through the
Walkers.

```python
self.context.synchronize(restore_walker)
```

## Step 6

Completed code in `rndwalk_6.py`.

Step 6 begins the logging of the colocation counts.

1. Add the dataclass that records the colocation count data.

```python
@dataclass
class ColocationLog:
    total_colocs: int = 0
    min_colocs: int = 0
    max_colocs: int = 0
```

2. Add colocation counting to the `Walker` in a `count_colocations` method:

```python
def count_colocations(self, grid, coloc_log: ColocationLog):
    # subtract self
    num_here = grid.get_num_agents(self.pt) - 1
    coloc_log.total_colocs += num_here
    if num_here < coloc_log.min_colocs:
        coloc_log.min_colocs = num_here
    if num_here > coloc_log.max_colocs:
        coloc_log.max_colocs = num_here
```

3. Add the call to `Walker.count_colocations` to the agent iteration in 
`step`.

```python
for walker in self.context.agents(shuffle=True):
    walker.count_colocations(self.grid, self.coloc_log)
    walker.walk(self.grid)
```

4. Create the `ColocationLog` in `Model.__init_` after the
agent creation loop.

```python
self.coloc_log = ColocationLog()
```

## Step 7

Completed code in `rndwalk_7.py`.

Step 7 completes the logging of co-location counts.

1. Add the logger creation to the bottom of `Model.__init__()`

```python
loggers = logging.create_loggers(self.coloc_log, op=MPI.SUM, names={'total_colocs': 'total'}, rank=rank)
loggers += logging.create_loggers(self.coloc_log, op=MPI.MIN, names={'min_colocs': 'min'}, rank=rank)
loggers += logging.create_loggers(self.coloc_log, op=MPI.MAX, names={'max_colocs': 'max'}, rank=rank)
self.data_set = logging.ReducingDataSet(loggers, comm, params['coloc_log_file'])
```

2. Add the initial colocation count logging for time 0 beneath
`self.data_set`

```python
# count the initial colocations at time 0 and log
for walker in self.context.agents():
    walker.count_colocations(self.grid, self.coloc_log)
self.data_set.log(0)
# clear the log counts
self.coloc_log.max_colocs = self.coloc_log.min_colocs = self.coloc_log.total_colocs = 0
```

3. Add the code to perform the logging every tick to `Model.step()`

```python
tick = self.runner.schedule.tick
self.data_set.log(tick)
# clear the log counts for the next tick
self.coloc_log.max_colocs = self.coloc_log.min_colocs = self.coloc_log.total_colocs = 0
```

4. Add the code to schedule `self.data_set.close` at model end underneath
`self.data_set = ...` in `Model.__init__()`

```python
self.runner.schedule_end_event(self.data_set.close)
```

Open the log file - `rndwalk/output/coloc_log.csv` to view 
the logged colocation counts.
