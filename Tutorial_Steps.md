# Tutorial Steps

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Repast/repast4py-tutorial.git/WSC_2022)

The tutorial builds a version of the Random Walk demonstration model.
The simulation itself consists of a number of agents moving at random around a two-dimensional grid 
and logging the current minimum and maximum distance from all the agent's 
starting points. In the final model, each timestep the following occurs:

1. All the agents (walkers) choose a random direction and move one unit in that direction.

2. All the agents determine their current distance from their starting point.

3. The minimum, and maxiumum distances are calculated across
all process ranks, and these values are logged as the minimum, and maximum distance
values.

The code consists of the following components:

1. A `Walker` class that implements the agent state and behavior.
2. A `Model` class responsible for initialization and managing the simulation.
3. A `restore_walker` function used to create an individual Walker when that Walker has moved (i.e., walked) to another process.
4. A `run` function that creates and starts the simulation.
5. An `if name == "main"` block that allows the simulation to be run from the command line.


The tutorial code consists of 4 files `python/rndwalk_[1-4].py`. Each file
contains commented lines that we will uncomment to progressively build the
model. Uncommenting the code in `rndwalk_1.py` yields `rndwalk_2.py` which
when uncommented will in turn yield `rndwalk_3.py` and so on. `rndwalk_final.py` 
implements the final model, and `rndwalk_commented.py` contains all the commented
code from each file, numbered accordingly. After completing each step,
you can run the model with:

1. python3 rndwalk_N.py random_walk.yaml 
2. mpirun -n 2 python3 rndwalk_N.py random_walk.yaml 

To see the effect of the additions in both the single and multiprocess
scenarios.

## Step 0

Code in `rndwalk_1.py`.

1. Open a terminal in the binder Jupyter Lab launcher, and do
the following:

```bash
$ cd rndwalk
$ cp rndwalk_0.py rndwalk.py
$ python rndwalk.py random_walk.yaml

{'random.seed': 42, 'stop.at': 50, 'walker.count': 1000, 'world.width': 2000, 'world.height': 2000, 'coloc_log_file': 'output/coloc_log.csv'}
```

The skeleton parses the parameters from a yaml file and prints them out.

Note that 
    * the `Walker` agent extends repast4py's `core.Agent`.
    * the `Model` contains a `SharedContext` and creates a population
    of `Walker` agents to add to that.

## Step 1

Code in `rndwalk_1.py`.

Step 1 adds the walker behavior using scheduled events.

1. Add a walk method to the Walker class, by uncommenting
it in the Walker class.

```python
class Walker(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)

    def walk(self):
        if self.id == 10:
            print(f'WALKER: {self.uid} walking')
```

2. In `Model.__init__()` uncomment the scheduling code:

```python
self.runner = schedule.init_schedule_runner(comm)
self.runner.schedule_repeating_event(1, 1, self.step)
self.runner.schedule_stop(params['stop.at'])
```

3. Uncomment the `Model.step` and `Model.start` methods:

```python
def step(self):
        for walker in self.context.agents():
            walker.walk()

def start(self):
    self.runner.execute()
```

4. Uncomment the start code in the `run` function

```python
def run(params: Dict):
    print(f'PARAMETERS: {params}')
    model = Model(MPI.COMM_WORLD, params)
    model.start()
```

Run it:

```bash
$ python rndwalk.py random_walk.yaml
(10, 0, 0)
(10, 0, 0)
...
$ mpirun -n 2 python rndwalk.py random_walk.yaml
(10, 0, 0)
(10, 0, 0)
...
(10, 0, 1)
(10, 0, 1)
...
```

Notice how in the second case we have 1K agents on each process rank (0 and 1).

## Step 2

Code in `rndwalk_2.py`.

Step 2 adds the grid to the model and the agent code for
walking around it.

1. Uncomment `Walker.__init__` and `Walker.walk`

```python
def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
        self.pt = pt

def walk(self, grid):
    # choose two elements from the OFFSET array
    # to select the direction to walk in the
    # x and y dimensions
    xy_dirs = random.default_rng.choice(Walker.OFFSETS, size=2)
    self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))
    if self.id == 10:
        print(f'{self.uid} walking at {self.pt}')
```

`__init__` has been updated to take an initial agent location as
a parameter. `walk` now takes the shared grid as a parameter and chooses
random distances to walk in the x and y dimensions on that grid.

2. In `Model.__init__` uncomment the code that creates the `SharedGrid`
and the code that adds agents to it.

```python
## create a bounding box equal to the size of the entire global world grid
box = space.BoundingBox(0, params['world.width'], 0, params['world.height'],
                        0, 0)
## create a SharedGrid of 'box' size with sticky borders that allows multiple agents
## in each grid location.
self.grid = space.SharedGrid(name='grid', bounds=box, 
                             borders=space.BorderType.Sticky,
                             occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
self.context.add_projection(self.grid)

rng = repast4py.random.default_rng
self.rank = comm.Get_rank()
for i in range(params['walker.count']):
    # get a random x,y location in the grid
    pt = self.grid.get_random_local_pt(rng)
    # create and add the walker to the context
    walker = Walker(i, self.rank, pt)
    self.context.add(walker)
    self.grid.move(walker, pt)
```

3. Uncomment the code in `Model.step` to walk the agents, but
now passing the grid.

```python
def step(self):
    for walker in self.context.agents():
        walker.walk(self.grid)
```

## Step 3

Code in `rndwalk_3.py`.

Step 3 adds synchronization. In step 2, walkers could walk off their
local grid into "nowhere". Step 3 adds cross-process synchronization
that will move agents between processes such that when an agent walks out
of its local area, it is moved to the process containing the area it has
walked into. 

1. Uncomment the new walking code in `Walker.walk`.

```python
self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))
if self.local_rank != self.uid_rank:
    print(f'{self.uid} walking at {self.pt} on rank {self.local_rank}')
```

This will print out the id of those Walkers that have moved to
a different rank.

2. Uncomment `Walker.save`.

```python
def save(self) -> Tuple:
    """Saves the state of this Walker as a Tuple.

    Returns:
       The saved state of this Walker.
    """
    return (self.uid, self.pt.coordinates)
```

This saves the state of the walker as a tuple.

3. Uncomment `walker_cache` and the `restore_walker` function.

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

The restore function is used to create `Walker` agents from
the tuple returned from `Walker.save`. The `walker_cache` is
used to cache agents so they don't have to reconstructed if
when they enter a new rank multiple times.

4. Uncomment the `synchronize` call in `Model.step`.

```python
self.context.synchronize(restore_walker)
```

`synchronize` will use the `restore_walker` function to
properly move agents between process ranks.

When running Step 3, increase the `stop.at` parameter so there
is enough iterations for agents to walk off their original ranks

## Step 4

Code in `rndwalk_4.py`.

Step 4 adds the distance logging to the model. The distance logging
will log the minimum and maximum current distance from an origin
over all the agents current distances.

1. Uncomment the `DistanceLog` dataclass.

```python
@dataclass
class DistanceLog:
    min_distance: float = 0
    max_distance: float = 0
```

Each rank will record its own minimum and maximum distance
in this dataclass.

2. Uncomment the `starting_pt` assignment in
`Walker.__init__`. 

```python
self.starting_pt = pt
```

3. Uncomment `Walker.distance`.

```python
def distance(self):
    return np.linalg.norm(self.starting_pt.coordinates -
                          self.pt.coordinates)
```
This returns the Euclidian distance between the Walker's starting point and its current point.

4. Uncomment the log initialization code in `Model.__init__`

```python
self.log = DistanceLog()
# loggers = logging.create_loggers(self.log, op=MPI.MIN, names={'min_distance': 'min'}, rank=self.rank)
# loggers += logging.create_loggers(self.log, op=MPI.MAX, names={'max_distance': 'max'}, rank=self.rank)
# self.data_set = logging.ReducingDataSet(loggers, comm, params['log.file'])
# self.runner.schedule_end_event(self.data_set.close)
```

A logger is responsible for recording data from one or more fields in a dataclass, and
applying a cross process reduction operation on those value(s) (e.g., summing across ranks)
to yield the final logged value(s). These loggers are passed to a `ReducingDataSet` which you
will use to log the data at specific times. Lastly, we need to close the data set when 
the model terminates in order to write any remaining data, and we do this with a scheduled
end event that calls `close` on the data set when the model ends.

5. Uncomment `Model.log_distance`.

```python
def log_distance(self, walker):
    distance = walker.distance()
    if distance < self.log.min_distance:
        self.log.min_distance = distance
    if distance > self.log.max_distance:
        self.log.max_distance = distance
```

This checks each Walker to see if it's current distance is a new
minimum or maximum. If so, then set the appropriate dataclass field.

6. Uncomment the new logging code in `Model.step`.

```python
self.log.max_distance = float('-inf')
self.log.min_distance = float('inf')

for walker in self.context.agents():
    walker.walk(self.grid)
    self.log_distance(walker)

tick = self.runner.schedule.tick
self.data_set.log(tick)
```

This resets the maximum and minimum dataclass fields for the next tick,
checks each `Walkers` distance by calling `log_distance` and then logs
the current values in the dataset by performing the cross-process reduction
and logging those values.

When you run this, the data will be logged to `output/distance_log.csv` file,
as specified in the yaml parameters. Opening that file, you should see 
something like:

```csv
tick,min,max
1,1.4142135623730951,1.4142135623730951
2,0.0,2.8284271247461903
3,1.0,4.242640687119285
4,0.0,5.656854249492381
5,1.4142135623730951,7.0710678118654755
...
```
Note that the data logging code will automatically create a new file name from
the original, if that file already exists. For example, if 
`output/distance_log.csv` exists then `output/distance_log_1.csv` will
be used. If that file existsn then `output/distance_log_2.csv` will be used,
and so on.
