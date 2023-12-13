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
    def __init__(self, jobid_add, jobid_rem): #Identifica os objetos que serão adicionados e removidos
        self.jobid_add = jobid_add
        self.jobid_rem = jobid_rem
    
    def __str__(self): #Representa objeto como str
        return "jobid_add: {}, jobid_rem: {}".format(self.jobid_add, self.jobid_rem)


# -----


class Solution:
    def __init__(self, problem, used, unused, t, ms, lb): # Inicializa a solução
        self.problem = problem
        self.used = used #Job usado
        self.unused = unused #Job não-usado
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
    
    # Always can add [for constructive methods]
    def _can_add(self, jobid):
        return True 
    
    # Always can swap [for local search methods]
    def _can_swap(self, jobid_add, jobid_rem):
        return True
    
    # Return all components that can be add
    def add_moves(self) -> Iterable[Component]:
        for jobid in self.unused:
            if self._can_add(jobid):
                yield Component(jobid)

    # Add, remove and swap movements
    def local_moves(self) -> Iterable[LocalMove]:
        for jobid in self.unused: # para adicionar
            if self._can_add(jobid):
                yield LocalMove(jobid, None)
        for jobid in self.unused: # para remover
            yield LocalMove(None, jobid)
        for jobid_add in self.unused: # para trocar
            for jobid_rem in self.used:
                if self._can_swap(jobid_add, jobid_rem):
                    yield LocalMove(jobid_add, jobid_rem)
    
    # Função auxiliar para ajudar na próxima função / adiciona um trabalho à solução
    def _add_job(self, jobid):
        prob = self.problem
        self.used.append(jobid)
        self.unused.remove(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.t[i] += prob.r[i][jobid] # o tempo de processamento é igual ao tempo do trabalho
            else: # se não for a primeira máquina
                self.t[i] = max(self.t[i], self.t[i-1]) + prob.r[i][jobid] # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, mais o tempo do trabalho
        self.ms = self.t[-1] # o makespan é o tempo da última máquina
    
    # Função auxiliar para ajudar na próxima função / remove um trabalho à solução
    def _remove_job(self, jobid):
        prob = self.problem
        self.used.remove(jobid)
        self.unused.add(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.t[i] -= prob.r[i][jobid] # o tempo de processamento é reduzido pelo tempo do trabalho
            else: # se não for a primeira máquina
                self.t[i] = max(self.t[i] - prob.r[i][jobid], self.t[i-1]) # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, menos o tempo do trabalho
        self.ms = self.t[-1] # o makespan é o tempo da última máquina
    
    # Adiciona um componente à solução e atualiza o lower bound
    def add(self, c: Component) -> None:
        self._add_job(c.jobid)
        self.lb = self._lb_update_add(c.jobid)

    def step(self, lmove: LocalMove) -> None:
        # TODO # This invalidates the lower bound
        # Apply a local move to the solution. This invalidates any previously generated components and local moves.
        if lmove.jobid_rem is not None:
            self._remove_job(lmove.jobid_rem)
            # TODO # self.lb = self._lb_update_rem(c.jobid_rem)
        if lmove.jobid_add is not None:
            self._add_job(lmove.jobid_add)
            # self.lb = self._lb_update_add(c.jobid_add)
        # se houver algo para retirar, retiramos, se houver para acrescentar, o fazemos
    
    # O incremento da função objetivo: diff entre os tempos dos trabalhos na última máquina [se local search]
    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        incr = (self.problem.r[-1][lmove.jobid_rem] if lmove.jobid_rem is not None else 0)\
              -(self.problem.r[-1][lmove.jobid_add] if lmove.jobid_add is not None else 0) 
        return incr

    # O incremento da função objetivo: diff entre os tempos dos trabalhos na última máquina [se construtivo]
    def objective_incr_add(self, c: Component) -> Optional[Objective]:
        return self.problem.r[-1][c.jobid]
    
    # Retorna o incremento do lower bound [se construtivo]
    def lower_bound_incr_add(self, c: Component) -> Optional[Objective]:
        lb = self._lb_update_add(c.jobid) # calcula o novo lower bound
        return lb - self.lb

    # Retorna atualização do lower bound [se construtivo]
    def _lb_update_add(self, jobid):
        prob = self.problem
        lb = 0 # inicializa o lower bound
        for i in range(prob.n): # para cada máquina
            lb = max(lb, self.t[i] + prob.r[i][jobid]) # o lower bound é o máximo entre o lower bound atual e o tempo de processamento na máquina atual, mais o tempo do trabalho
        return lb


# -----


class Problem:
    def __init__(self, t, r):
        self.t = t
        self.r = r
        self.n = len(r)  # Number of jobs
        self.m = len(r[0]) if r else 0  # Number of machines, assuming r is not empty
        if not t or not r:
            self.lb = 0
        else:
            self.lb = max(sum(t), sum([max(r[i]) for i in range(self.n)]))

    def __str__(self):
        s = [str(self.n), str(self.m)] + list(map(str, self.t))
        for i in range(self.n):
            s.append(" ".join(map(str, self.r[i])))
        return "\n ".join(s)

    @classmethod
    def from_textio(cls, textio: TextIO) -> Problem:
        try:
            m, n = map(int, textio.readline().split()[:2])  # Read only the first two numbers
            r = [list(map(int, textio.readline().split())) for _ in range(n)]
            return cls([], r)  # Pass an empty t since it's not needed
        except (ValueError, IndexError):
            raise ValueError("Invalid input format")
    
    # Retorna a solução vazia
    def empty_solution(self) -> Solution:
        return Solution(self, [], set(range(self.m)), [0] * self.n, 0, self.lb)


# -------------------------------------------------


if __name__ == '__main__':
    # Read the problem from stdin
    problem = Problem.from_textio(sys.stdin)

    # Test your functions here
    print("Problem:")
    print(problem)

    # Example of creating a solution and testing some methods
    solution = Solution(problem, [], set(range(problem.m)), [0] * problem.n, 0, problem.lb)
    print("\nInitial Solution:")
    print(solution)

    for move in solution.add_moves():
        print("\nAdding job:", move)
        # Create a copy of the solution to keep the state across iterations
        new_solution = solution.copy()
        new_solution.add(move)
        # Printa demais informações
        print("Updated Solution:")
        print(new_solution)
        print("Is Feasible:", new_solution.is_feasible())
        print("Objective Value:", new_solution.objective())
        print("Lower Bound:", new_solution.lower_bound())