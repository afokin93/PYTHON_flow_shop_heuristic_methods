from typing import List
from Flow import Problem, Solution, Component
import sys 

def greedy_construction(problem: Problem) -> List[int]:
    solution = Solution(problem, [], set(range(problem.n)), [0] * problem.m, 0, problem.lb)
    unused_jobs = list(range(problem.n))

    while unused_jobs:
        best_job, best_incr = None, float('inf')
        for jobid in unused_jobs:
            incr = solution.lower_bound_incr_add(Component(jobid))
            if incr < best_incr:
                best_job, best_incr = jobid, incr

        if best_job is not None:
            solution.add(Component(best_job))
            unused_jobs.remove(best_job)

    return solution.used

if __name__ == '__main__':
    # Read the problem from stdin
    problem = Problem.from_textio(sys.stdin)

    # Solve the problem using the greedy construction heuristic
    job_sequence = greedy_construction(problem)

    # Print the final job sequence
    print("\nFinal Job Sequence:")
    print(" ".join(map(str, job_sequence)))
