from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(f"Hello, World! I am rank {rank} of {size}")
if rank == 0:
    message = [1, 2, 3]
else:
    message = None

message = comm.bcast(message, root=0)
print(f"{rank} received {message}")
