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
        # TODO: Implement this method to calculate the objective function
        pass

    def add_moves(self) -> Iterable[Component]: 
        # TODO: Implement this method to generate possible moves (job swaps)
        pass

    def add(self, c: Component) -> None:
        # TODO: Implement this method to add a job to the sequence
        pass




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
        num_jobs, num_machines = map(int, f.readline().split())
        jobs = []
        for i in range(num_jobs):
            processing_times = list(map(int, f.readline().split()))
            assert len(processing_times) == num_machines
            jobs.append(Component(i, processing_times))
        return cls(jobs)

    def initial_solution(self) -> Solution:
        # TODO: Implement this method to generate an initial solution
        pass


if __name__ == '__main__':
    problem = Problem.from_textio(sys.stdin)
    print(problem)
    s = problem.initial_solution()
    print(s.sequence)

    components = list(s.add_moves())

    