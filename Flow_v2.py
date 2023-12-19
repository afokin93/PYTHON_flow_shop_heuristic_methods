from __future__ import annotations
from typing import TextIO, Optional, Any
import sys, random


# -------------------------------------------------


class Component: #O que é uma componente neste problema? --> Um job!
    def __init__(self, jobid):
        self.jobid = jobid #Inicializa o id do trabalho
    
    def __str__(self):
        return "job_id:" + str(self.jobid) #Devolve uma string como representação do job
    
    def id(self):
        return self.jobid #Devolve, em memória, o job_id que a componente representa


# -----


class LocalMove:
    def __init__(self, job_a, job_b): #Identifica os objetos que terão swap feito entre si
        self.job_a = job_a
        self.job_b = job_b

    def __str__(self): #Representa objeto como str
        return "job_a: {}, job_b: {}".format(self.job_a, self.job_b)


# -----


class Solution:
    def __init__(self, problem, used, unused, t, ms, lb): # Inicializa a solução
        self.problem = problem
        self.used = list(used) #Job usado
        self.unused = list(unused) #Job não-usado
        self.t = t #Tempo de processamento em cada máquina
        self.ms = ms #Makespan
        self.lb = lb #Lower bound
    
    def __str__(self):
        return " ".join([str(u) for u in self.used])
    
    def copy(self) -> Solution:
        return self.__class__(self.problem,
                              self.used.copy(),
                              self.unused.copy(),
                              self.t.copy(),
                              self.ms,
                              self.lb)
    
    # Return whether the solution is feasible or not --> Always is
    def is_feasible(self) -> bool: 
        return True
    
    # Return the objective value for this solution if defined, otherwise should return None
    def objective(self) -> Optional[Objective]:
        return self.ms 
    
    # Return the lower bound value for this solution if defined, otherwise return None
    def lower_bound(self) -> Optional[Objective]:
        return self.lb 
    
    # Always can swap [for local search methods]
    def _can_swap(self, job_a, job_b):
        return True

    # Primeira função de movimentos --> 1670
#    def local_moves(self) -> Iterable[LocalMove]:
#        used_list = list(self.used)
#        for i in range(len(used_list)):
#            for j in range(i + 1, len(used_list)):
#                job_a, job_b = used_list[i], used_list[j]
#                if self._can_swap(job_a, job_b):
#                    yield LocalMove(job_a, job_b)

    # Segunda função de movimentos --> 1637
#    def generate_swap_moves(self):
#        swap_moves = []
#        for i in range(len(self.used) - 1):
#            move_pairs = (self.used[i], self.used[i + 1])
#            swap_moves.append(move_pairs)
#        return swap_moves
#    def local_moves(self):
#        swap_moves = self.generate_swap_moves()
#        for move_pairs in swap_moves:
#            job_a, job_b = move_pairs
#            if self._can_swap(job_a, job_b):
#                yield LocalMove(job_a, job_b)

    # Terceira função de movimentos --> 1514
    def generate_swap_moves(self, start_sequence):
        swap_moves = []
        for i in range(len(start_sequence)):
            for j in range(len(start_sequence)):
                move_pairs = (start_sequence[i], start_sequence[j])
                swap_moves.append(move_pairs)
        return swap_moves
    def local_moves(self):
        seen_sequences = set()  # Keep track of unique sequences
        # Iteration with the original sequence
        swap_moves = self.generate_swap_moves(start_sequence=self.used)
        for move_pairs in swap_moves:
            job_a, job_b = move_pairs
            if self._can_swap(job_a, job_b):
                new_used = self.step(LocalMove(job_a, job_b))
                if new_used is not None and tuple(new_used) not in seen_sequences:
                    seen_sequences.add(tuple(new_used))
                    yield LocalMove(job_a, job_b)

    # Função auxiliar para ajudar na próxima função / troca dois jobs na solução completa
    def step(self, lmove):
        prob = self.problem
        # Extrai os trabalhos da LocalMove
        job_a = lmove.job_a
        job_b = lmove.job_b
        # Create a copy of the set before modification
        new_used = list(job_b if jobid == job_a else (job_a if jobid == job_b else jobid) for jobid in self.used)
        #print(new_used)
        # Recalcula o makespan
        t = [0] * prob.n
        for jobid in new_used:
            for i in range(prob.n):
                if i == 0:
                    t[i] += prob.r[i][jobid]
                else:
                    t[i] = max(t[i], t[i - 1]) + prob.r[i][jobid]

        # Update the state after making changes
        self.used = new_used
        self.t = t
        self.ms = t[-1]
        lb = self._lb_update_local(jobid) # calcula o novo lower bound
        print(self.ms, lb)

    # O incremento da função objetivo: diff entre os tempos dos trabalhos na última máquina [se local search]
    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        incr = (self.problem.r[-1][lmove.job_b] if lmove.job_b is not None else 0)\
              -(self.problem.r[-1][lmove.job_a] if lmove.job_a is not None else 0)
        return incr

    # Retorna o incremento do lower bound [se local search]
    def lower_bound_incr_local(self, c: Component) -> Optional[Objective]:
        lb = self._lb_update_local(c.jobid) # calcula o novo lower bound
        print(lb)
        return lb - self.lb

    # Retorna atualização do lower bound [se local search]
    def _lb_update_local(self, jobid):
        prob = self.problem
        lb = self.ms
        #TODO abaixo
        for i in range(prob.n): # para cada máquina
            lb = max(lb, self.t[i] + prob.r[i][jobid]) # o lower bound é o máximo entre o lower bound atual e o tempo de processamento na máquina atual, mais o tempo do trabalho
        return lb


# -----


class Problem:
    def __init__(self, t, r):
        self.t = t # Tempo de processamento de cada máquina
        self.r = r # Job
        self.n = len(r)  # Number of jobs
        self.m = len(r[0]) if r else 0  # Number of machines, assuming r is not empty
        if not t or not r:
            self.lb = 0
        else:
            self.lb = max(sum(t), sum([max(r[i]) for i in range(self.n)]))

    def __str__(self):
        s = [str(self.n), str(self.m)] + list(map(str, self.t))
        for i in range(self.n):
            s.append("".join(map(str, self.r[i])))
        return "\n ".join(s)

    @classmethod
    def from_textio(cls, textio: TextIO) -> Problem:
        try:
            m, n = map(int, textio.readline().split()[:2])  # Read only the first two numbers
            r = [list(map(int, textio.readline().split())) for i in range(n)]
            return cls([], r)  # Pass an empty t since it's not needed
        except (ValueError, IndexError):
            raise ValueError("Invalid input format")
    
    # Retorna a solução vazia [para procura local]
    def initial_solution(self) -> Solution:
        return Solution(self, list(range(self.m)), [], self.t, self.lb, self.lb)

# self.problem = problem
# self.used = used #Job usado
# self.unused = unused #Job não-usado
# self.t = t #Tempo de processamento em cada máquina
# self.ms = ms #Makespan
# self.lb = lb #Lower bound

# -------------------------------------------------


if __name__ == '__main__':
    # Read the problem from stdin
    problem = Problem.from_textio(sys.stdin)

    # Test your functions here
    print("Problem:")
    print(problem)

    # Example of creating a solution and testing some methods
    solution = problem.initial_solution()
    print("\nSolution Zero - Initial for Local Search:")
    print(solution.ms)
    print("\n")

    for move in solution.local_moves():
        #print("\nAdding job:", move)
        # Create a copy of the solution to keep the state across iterations
        new_solution = solution.copy()
        new_solution.step(move)
        # Print das demais informações
        #print("Updated Solution:")
        print(new_solution)
        #print(move)
        #print("Is Feasible:", new_solution.is_feasible())
        #print("Objective Value:", new_solution.objective())
        #print("Lower Bound:", new_solution.lower_bound())
        incr = new_solution.objective_incr_local(move)
        print("Increment:", incr)