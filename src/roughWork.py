from mpi4py import MPI
import time

def compute_square_root(start, end):
    result = 0
    for i in range(start, end + 1):
        result += i ** 0.5
    return result

def run_computation(num_processors):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    limit = 10000000

    if rank == 0:
        print(f"\nNumber of processors: {num_processors}")

    comm.barrier()  # Synchronize all processes before starting the computation

    start_time = time.time()
    local_start = 1 + rank * (limit // size)
    local_end = (rank + 1) * (limit // size) if rank != size - 1 else limit

    local_result = compute_square_root(local_start, local_end)
    total_result = comm.reduce(local_result, op=MPI.SUM, root=0)

    elapsed_time = time.time() - start_time
    max_elapsed_time = comm.reduce(elapsed_time, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"Time taken on {num_processors} processors: {max_elapsed_time:.6f} seconds")
        return num_processors, max_elapsed_time

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"{rank=}")

    if rank == 0:
        results = []

        for num_processors in [2, 4, 8, 16]:  # You can add more processors as needed
            result = comm.scatter([run_computation(num_processors) for _ in range(comm.Get_size())], root=0)
            results.append(result)

        print("\nResults:")
        print("Number of Processors | Time Taken (seconds)")
        print("-------------------- | ---------------------")
        for result in results:
            print(f"{result[0]:^20} | {result[1]:^20.6f}")
