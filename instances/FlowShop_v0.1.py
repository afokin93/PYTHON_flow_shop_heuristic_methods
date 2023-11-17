from __future__ import annotations
from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable
import sys, random

Objective = Any


# ------------------------------------------------------------------------------------------------------- #


class Component: #used to represent a job
    def __init__(self, job_id: int, processing_times: list[int]):
        self.job_id = job_id
        self.processing_times = processing_times

#job = Component(1, [2, 3, 4])
#print(job.job_id)  # Output: 1
#print(job.processing_times)  # Output: [2, 3, 4]


# ------------------------------------------------------------------------------------------------------- #


class Solution:
    def __init__(self, problem, sequence):
        self.problem = problem
        self.sequence = sequence  # sequence of jobs


    def __str__(self):
        return "->".join([str(job.job_id) for job in self.sequence])


    def is_feasible(self) -> bool:
        """ Return whether the solution is feasible or not """
        return len(self.sequence) == self.problem.num_jobs


    def objective(self) -> Optional[Objective]:
        if not self.is_feasible():
            return None
        # Initialize a matrix to hold the completion times of each job on each machine
        completion_times = [[0 for _ in range(self.problem.num_machines)] for _ in range(self.problem.num_jobs)]
    
        # Calculate the completion time of the first job on each machine
        for machine in range(self.problem.num_machines):
            if machine == 0:
                completion_times[0][machine] = self.sequence[0].processing_times[machine]
            else:
                completion_times[0][machine] = completion_times[0][machine - 1] + self.sequence[0].processing_times[machine]
        # Calculate the completion time of each job on the first machine
        for job in range(1, self.problem.num_jobs):
            completion_times[job][0] = completion_times[job - 1][0] + self.sequence[job].processing_times[0]
        
        # Calculate the completion time of each job on each machine
        for job in range(1, self.problem.num_jobs):
            for machine in range(1, self.problem.num_machines):
                completion_times[job][machine] = max(completion_times[job - 1][machine], completion_times[job][machine - 1]) + self.sequence[job].processing_times[machine]
        
        # The makespan is the completion time of the last job on the last machine
        makespan = completion_times[-1][-1]
        return makespan


    def lower_bound(self) -> Optional[Objective]:
        # Calculate the load on each machine
        loads = [0 for _ in range(self.problem.num_machines)]
        for job in self.sequence:
            for machine in range(self.problem.num_machines):
                if machine == 0 or job.job_id == 0:
                    loads[machine] += job.processing_times[machine]
                else:
                    loads[machine] = max(loads[machine], loads[machine - 1]) + job.processing_times[machine]

        # The lower bound is the maximum load on any machine
        return max(loads)


    def lower_bound_increment(self, job: Component, machine: int) -> Optional[Objective]:
        # Calculate the increment in the load on the machine
        increment = job.processing_times[machine]

        # The increment in the lower bound is the increment in the load
        return increment


    def add_moves(self) -> Iterable[Component]:
    # Generate all pairs of jobs in the sequence
        for i in range(len(self.sequence)):
            for j in range(i + 1, len(self.sequence)):
                # Yield a tuple representing a swap of jobs i and j
                yield (i, j)


    def add(self, job: Component) -> None:
    # Add the job to the end of the sequence
        self.sequence.append(job)


# ------------------------------------------------------------------------------------------------------- #


class Problem:
    def __init__(self, jobs: list[Component], num_machines: int):
        self.jobs = jobs  # list of jobs
        self.num_jobs = len(jobs)
        self.num_machines = num_machines  # Adicionando o número de máquinas como um atributo


    def __str__(self):
        s = ["{} jobs, {} machines".format(self.num_jobs, self.num_machines)]
        for job in self.jobs:
            s.append("Job {}: {}".format(job.job_id, job.processing_times))
        return "\n".join(s)


    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        line = f.readline().split()
        num_jobs, num_machines = int(line[0]), int(line[1])
        print(f"Number of jobs: {num_jobs}, Number of machines: {num_machines}")
        jobs = []

        for i in range(num_machines):
            processing_times = list(map(int, f.readline().split()))
            print("Processing times for each job in machine {}: {}".format(i, processing_times))
            #print("Expected number of machines: {}".format(num_machines))
            print(processing_times)
            #print("Actual number of machines in processing_times: {}".format(len(processing_times)))
            #assert len(processing_times) == num_machines
            jobs.append(Component(i, processing_times))
        return cls(jobs, num_machines)


    def initial_solution(self) -> Solution:
        # Criar uma sequência inicial com base no número de máquinas
        initial_sequence = [Component(i, [0] * self.num_machines) for i in range(self.num_jobs)]
            # Adicionar os valores específicos para cada processing_times
        return Solution(self, initial_sequence)


    def empty_solution(self) -> Solution:
        empty_solution = []
        return Solution(self, [])


# ------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    # Test the Problem.from_textio method
    problem = Problem.from_textio(sys.stdin)
    print("Problem:")
    print(problem)

    # Test the Empty solution
    s0 = problem.empty_solution()
    print("\nEmpty solution:")
    print(s0.sequence)

    # Test the Solution object and its methods
    s1 = problem.initial_solution()
    print("\nInitial Solution:")
    print(s1.sequence)

    # Test the Solution.is_feasible method
    print("\nIs Feasible:", s1.is_feasible())

    # Test the Solution.objective method
    print("Objective:", s1.objective())

    # Test the Solution.lower_bound method
    print("Lower Bound:", s1.lower_bound())

    # Test the Solution.lower_bound_increment method
    if s1.is_feasible():
        job = s1.sequence[0]  # Choose a job from the sequence
        machine = 0  # Choose a machine
        print("Lower Bound Increment:", s1.lower_bound_increment(job, machine))

    # Test the Solution.add_moves method
    print("\nPossible Moves:")
    for move in s1.add_moves():
        print(move)

    # Test the Solution.add method
    new_job = Component(99, [1, 2, 3])  # Create a new job
    s1.add(new_job)
    print("\nUpdated Solution:")
    print(s1.sequence)    