from __future__ import annotations
from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable
import sys, random

Objective = Any

class Component: #used to represent a job
    def __init__(self, job_id: int, processing_times: list[int]):
        self.job_id = job_id
        self.processing_times = processing_times

#job = Component(1, [2, 3, 4])
#print(job.job_id)  # Output: 1
#print(job.processing_times)  # Output: [2, 3, 4]

class Solution:
    def __init__(self, problem, sequence):
        self.problem = problem
        self.sequence = sequence  # sequence of jobs

    def __str__(self):
        return "->".join([str(job.job_id) for job in self.sequence])

    def is_feasible(self) -> bool:
        """ Return whether the solution is feasible or not """
        return len(self.sequence) == len(self.problem.components)

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





class Problem:
    def __init__(self, jobs: list[Component]):
        self.jobs = jobs  # list of jobs
        self.num_jobs = len(jobs)
        self.num_machines = len(jobs[0].processing_times)

    def __str__(self):
        s = ["{} jobs, {} machines".format(self.num_jobs, self.num_machines)]
        for job in self.jobs:
            s.append("Job {}: {}".format(job.job_id, job.processing_times))
        return "\n".join(s)

    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        num_jobs, num_machines, *_ = map(int, f.readline().split())
        print(f"Number of jobs: {num_jobs}, Number of machines: {num_machines}")
        jobs = []
        
        # Skip the additional values in the first line
        f.readline()
        
        for i in range(num_jobs):
            processing_times = list(map(int, f.readline().split()))
            print(f"Processing times for Job {i}: {processing_times}")
            assert len(processing_times) == num_machines
            jobs.append(Component(i, processing_times))
        return cls(jobs)


    def initial_solution(self) -> Solution:
        # Create a solution with the jobs in their initial order
        initial_sequence = self.jobs.copy()
        return Solution(self, initial_sequence)



if __name__ == '__main__':
    # Test the Problem.from_textio method
    problem = Problem.from_textio(sys.stdin)
    print("Problem:")
    print(problem)

    # Test the Solution object and its methods
    s = problem.initial_solution()
    print("\nInitial Solution:")
    print(s.sequence)

    # Test the Solution.is_feasible method
    print("\nIs Feasible:", s.is_feasible())

    # Test the Solution.objective method
    print("Objective:", s.objective())

    # Test the Solution.lower_bound method
    print("Lower Bound:", s.lower_bound())

    # Test the Solution.lower_bound_increment method
    if s.is_feasible():
        job = s.sequence[0]  # Choose a job from the sequence
        machine = 0  # Choose a machine
        print("Lower Bound Increment:", s.lower_bound_increment(job, machine))

    # Test the Solution.add_moves method
    print("\nPossible Moves:")
    for move in s.add_moves():
        print(move)

    # Test the Solution.add method
    new_job = Component(99, [1, 2, 3])  # Create a new job
    s.add(new_job)
    print("\nUpdated Solution:")
    print(s.sequence)

    