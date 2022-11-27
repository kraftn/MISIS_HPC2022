import argparse
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


class CellularAutomaton:
    def __init__(self, automaton_rule: int = 110, is_periodic: bool = True):
        """
        Параллельный одномерный клеточный автомат.

        :param automaton_rule: код Вольфрама
        :param is_periodic: использовать периодические граничные условия
        """
        rules = bin(automaton_rule)[2:]
        rules = '0' * (8 - len(rules)) + rules
        self.table = [int(rule) for rule in rules[::-1]]
        self.is_periodic = is_periodic
        self.comm = MPI.COMM_WORLD
        self.process_rank = self.comm.Get_rank()

    def compute(self, n_times: int, n_cells: t.Optional[int] = None, initial_state: t.Optional[np.ndarray] = None):
        """
        Вычислить матрицу состояний.

        :param n_times: количество точек по времени
        :param n_cells: количество ячеек
        :param initial_state: начальное состояние
        :return: матрица состояний размером n_times x n_cells
        """
        n_processes = self.comm.Get_size()
        if initial_state is not None:
            n_cells = initial_state.shape[0]
        if n_cells // n_processes == 0:
            n_processes = n_cells
            if self.process_rank >= n_processes:
                return

        states = self._get_process_initial_states(n_times, n_cells, initial_state, n_processes)
        self._run_time_steps(states, n_processes)
        states = self._gather_states(states, n_processes)

        return states

    def show_evolution_image(self, n_times: int, n_cells: t.Optional[int] = None,
                             initial_state: t.Optional[np.ndarray] = None):
        """
        Построить и вывести изображение эволюции.

        :param n_times: количество точек по времени
        :param n_cells: количество ячеек
        :param initial_state: начальное состояние
        """
        states = self.compute(n_times, n_cells, initial_state)
        if self.process_rank == 0:
            figure = plt.figure(figsize=(6, 6))
            figure.gca().axis('off')
            plt.imshow(states, cmap='Greys')
            plt.tight_layout()
            plt.show()

    def conduct_experiments(self, n_experiments: int):
        """
        Вычислить среднее время работы метода ``compute`` для ``n_experiments`` запусков.

        :param n_experiments: количество запусков метода compute
        """
        time = MPI.Wtime()
        for _ in range(n_experiments):
            self.compute(n_times=2000, n_cells=2000)
        time = (MPI.Wtime() - time) / n_experiments

        if self.process_rank == 0:
            n_processes = self.comm.Get_size()
            if not os.path.exists('results'):
                os.mkdir('results')
            with open('results/time', 'a', encoding='utf-8') as file:
                file.write(f'{n_processes} {time}\n')

    def show_time_plot(self):
        """
        Вывести график зависимости среднего времени работы метода ``compute`` от количества запущенных процессов.
        """
        if self.process_rank == 0:
            with open('results/time', 'r', encoding='utf-8') as file:
                data = file.readlines()

            results = []
            for line in data:
                line_arr = line.split(' ')
                n_processes = int(line_arr[0])
                mean_time = float(line_arr[1][:-2])
                results.append((n_processes, mean_time))
            results.sort()

            plt.figure(figsize=(8, 5))
            plt.plot(*zip(*results))
            plt.xlabel('Количество процессов')
            plt.ylabel('Время, с')
            plt.title('Зависимость среднего времени работы программы от количества процессов')
            plt.tight_layout()
            plt.show()

    def _get_process_initial_states(self, n_times: int, n_cells: int, initial_state: np.ndarray, n_processes: int):
        """
        Инициализировать матрицу состояний для каждого процесса.

        :param n_times: количество точек по времени
        :param n_cells: количество ячеек
        :param initial_state: начальное состояние клеточного автомата
        :param n_processes: количество активных процессов
        :return: матрица состояний для процесса с инициализированным начальным состоянием
        """
        n_process_cells = n_cells // n_processes + 2
        if self.process_rank < n_cells % n_processes:
            n_process_cells += 1
        states = np.empty((n_times, n_process_cells), dtype=np.int8)

        if initial_state is not None:
            start_index = self.process_rank * n_cells // n_processes
            start_index += min(self.process_rank, n_cells % n_processes)
            end_index = start_index + n_process_cells - 2
            states[0, 1:-1] = initial_state[start_index:end_index]
        else:
            states[0] = np.random.randint(0, 2, size=states[0].shape, dtype=np.int8)

        return states

    def _run_time_steps(self, states: np.ndarray, n_processes: int):
        """
        Выполнить шаги по времени.

        :param states: матрица состояний процесса
        :param n_processes: количество активных процессов
        """
        for i_time, prev_state in enumerate(states[:-1]):
            curr_state = states[i_time + 1]

            self._exchange_data(prev_state, n_processes)
            for i_cell in range(1, prev_state.shape[0] - 1):
                key = prev_state[i_cell - 1] * 4 + prev_state[i_cell] * 2 + prev_state[i_cell + 1]
                curr_state[i_cell] = self.table[key]

            if not self.is_periodic and self.process_rank == 0:
                curr_state[1] = prev_state[1]
            if not self.is_periodic and self.process_rank == n_processes - 1:
                curr_state[-2] = prev_state[-2]

    def _exchange_data(self, state: np.ndarray, n_processes: int):
        """
        Выполнить пересылку "клеток-призраков".

        :param states: матрица состояний процесса
        :param n_processes: количество активных процессов
        """
        left_neighbor = (self.process_rank - 1) % n_processes
        right_neighbor = (self.process_rank + 1) % n_processes
        send1 = self.comm.isend(state[1].item(), dest=left_neighbor, tag=1)
        send2 = self.comm.isend(state[-2].item(), dest=right_neighbor, tag=2)
        state[0] = self.comm.irecv(source=left_neighbor, tag=2).wait()
        state[-1] = self.comm.irecv(source=right_neighbor, tag=1).wait()
        send1.wait()
        send2.wait()

    def _gather_states(self, states: np.ndarray, n_processes: int):
        """
        Собрать результаты расчётов на нулевом процессе.

        :param states: матрица состояний процесса
        :param n_processes: количество активных процессов
        :return: матрица состояний клеточного автомата
        """
        if self.process_rank != 0:
            states = np.ascontiguousarray(states[:, 1:-1])
            self.comm.ssend(states.shape, dest=0, tag=3)
            self.comm.Ssend([states, MPI.INT8_T], dest=0, tag=4)
            return
        else:
            states = states[:, 1:-1]
            for i_process in range(1, n_processes):
                neighbor_states_shape = self.comm.recv(source=i_process, tag=3)
                neighbor_states = np.empty(neighbor_states_shape, dtype=np.int8)
                self.comm.Recv([neighbor_states, MPI.INT8_T], source=i_process, tag=4)
                states = np.hstack([states, neighbor_states])
            return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['image', 'experiments', 'plot'])
    parser.add_argument('--automaton_rule', required=False, type=int, default=110)
    parser.add_argument('--periodic', required=False, action='store_true')
    parser.add_argument('--random', required=False, action='store_true')
    parser.add_argument('--n_experiments', required=False, type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'image':
        automaton = CellularAutomaton(automaton_rule=args.automaton_rule, is_periodic=args.periodic)
        n_cells = n_times = 1000
        if not args.random:
            initial_state = np.zeros(n_cells, dtype=np.int8)
            initial_state[n_cells // 2] = 1
            automaton.show_evolution_image(n_times, initial_state=initial_state)
        else:
            automaton.show_evolution_image(n_times, n_cells=n_cells)
    elif args.mode == 'experiments':
        automaton = CellularAutomaton()
        automaton.conduct_experiments(args.n_experiments)
    elif args.mode == 'plot':
        automaton = CellularAutomaton()
        automaton.show_time_plot()
