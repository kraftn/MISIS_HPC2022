import random

from mpi4py import MPI


def main():
    name_template = 'Process {}'

    comm = MPI.COMM_WORLD
    process_rank = comm.Get_rank()
    size = comm.Get_size()

    if process_rank == 0:
        names = [(process_rank, name_template.format(process_rank))]
        dest = random.randint(1, size - 1)
        comm.ssend(names, dest=dest, tag=1)

        names = comm.recv(tag=1)
        for rank, name in names:
            print(f'Name of process with rank {rank} is \"{name}\"')
    else:
        names = comm.recv(tag=1)
        if len(names) == size - 1:
            dest = 0
        else:
            free_ranks = list(range(size))
            for rank, _ in names:
                free_ranks.remove(rank)
            free_ranks.remove(process_rank)
            dest = random.choice(free_ranks)

        names.append((process_rank, name_template.format(process_rank)))
        comm.ssend(names, dest=dest, tag=1)


if __name__ == '__main__':
    main()
