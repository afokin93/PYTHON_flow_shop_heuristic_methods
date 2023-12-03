from __future__ import annotations
from typing import TextIO, Optional, Any
import sys, random

# -----

class Component: # o que é uma componente neste problema? > um trabalho
    def __init__(self, jobid):
        self.jobid = jobid # inicializa o id do trabalho
    
    def __str__(self):
        return "job_id:" + str(self.jobid) # devolve uma string como representação do trabalho
    
    def id(self):
        return self.jobid

# -----

class LocalMove: # par de valores, quando for None não é para acontecer
    def __init__(self, jobid_add, jobid_rem):
        self.jobid_add = jobid_add
        self.jobid_rem = jobid_rem
    
    def __str__(self):
        return "jobid_add: {}, jobid_rem: {}".format(self.jobid_add, self.jobid_rem)

# -----

class Solution:
    def __init__(self, problem, used, unused, q, p, lb): # inicializa a solução
        # init serve para embrulhar as variáveis
        self.problem = problem
        self.used = used # trabalho usado
        self.unused = unused # trabalho não usado
        self.q = q # tempo de processamento em cada máquina
        self.p = p # makespan
        self.lb = lb # lower bound
    
    def __str__(self):
        return " ".join([str(u) for u in self.used])
    
    def copy(self) -> Solution:
        return self.__class__(self.problem,
                              self.used.copy(),
                              self.unused.copy(),
                              self.q.copy(),
                              self.p,
                              self.lb)
    
    def is_feasible(self) -> bool:
        return True # sempre # Return whether the solution is feasible or not
    
    def objective(self) -> Optional[Objective]:
        return self.p # Return the objective value for this solution if defined, otherwise should return None
    
    def lower_bound(self) -> Optional[Objective]:
        return self.lb # Return the lower bound value for this solution if defined, otherwise return None
    
    def _can_add(self, jobid):
        return True # sempre se pode adicionar um trabalho
    
    def _can_swap(self, jobid_add, jobid_rem):
        return True # sempre se pode trocar dois trabalhos
    
    def add_moves(self) -> Iterable[Component]: # mostra todos os componentes que podem ser adicionados à solução
        # Return an iterable (generator, iterator, or iterable object) over all components that can be added to the solution
        for jobid in self.unused:
            if self._can_add(jobid):
                yield Component(jobid) # yield: objeto que guarda resultados de funções
    
    def local_moves(self) -> Iterable[LocalMove]: # movimentos de adicionar, remover e de trocar
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        for jobid in self.unused: # para adicionar
            if self._can_add(jobid):
                yield LocalMove(jobid, None)
        for jobid in self.unused: # para remover
            yield LocalMove(None, jobid)
        for jobid_add in self.unused: # para trocar
            for jobid_rem in self.used:
                if self._can_swap(jobid_add, jobid_rem):
                    yield LocalMove(jobid_add, jobid_rem)
    
    def _add_job(self, jobid): # função auxiliar para ajudar na próxima função / adiciona um trabalho à solução
        prob = self.problem
        self.used.append(jobid)
        self.unused.remove(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.q[i] += prob.r[i][jobid] # o tempo de processamento é igual ao tempo do trabalho
            else: # se não for a primeira máquina
                self.q[i] = max(self.q[i], self.q[i-1]) + prob.r[i][jobid] # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, mais o tempo do trabalho
        self.p = self.q[-1] # o makespan é o tempo da última máquina
    
    def _remove_job(self, jobid): # função auxiliar para ajudar na próxima função / remove um trabalho à solução
        prob = self.problem
        self.used.remove(jobid)
        self.unused.add(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.q[i] -= prob.r[i][jobid] # o tempo de processamento é reduzido pelo tempo do trabalho
            else: # se não for a primeira máquina
                self.q[i] = max(self.q[i] - prob.r[i][jobid], self.q[i-1]) # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, menos o tempo do trabalho
        self.p = self.q[-1] # o makespan é o tempo da última máquina
    
    def add(self, c: Component) -> None: # adiciona um componente à solução
        self._add_job(c.jobid)
        self.lb = self._lb_update_add(c.jobid) # atualiza o lower bound
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
    
    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        return (self.problem.r[-1][lmove.jobid_rem] if lmove.jobid_rem is not None else 0) - (self.problem.r[-1][lmove.jobid_add] if
        lmove.jobid_add is not None else 0) # o incremento da função objetivo é a diferença entre os tempos dos trabalhos na última máquina
    
    def objective_incr_add(self, c: Component) -> Optional[Objective]:
        """Return the objective value increment resulting from adding a component.
        If the objective value is not defined after adding the component return None."""
        return self.problem.r[-1][c.jobid] # o incremento da função objetivo é o tempo do trabalho na última máquina
    
    def lower_bound_incr_add(self, c: Component) -> Optional[Objective]:
        """Return the lower bound increment resulting from adding a component.
        If the lower bound is not defined after adding the component, return None."""
        lb = self._lb_update_add(c.jobid) # calcula o novo lower bound
        return lb - self.lb # retorna a diferença entre o novo e o antigo lower bound
    
    def _lb_update_add(self, jobid): # função auxiliar para calcular o lower bound após adicionar um trabalho
        prob = self.problem
        lb = 0 # inicializa o lower bound
        for i in range(prob.n): # para cada máquina
            lb = max(lb, self.q[i] + prob.r[i][jobid]) # o lower bound é o máximo entre o lower bound atual e o tempo de processamento na máquina atual mais o tempo do trabalho
        return lb # retorna o lower bound

# -----
class Problem:
    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.n = len(q)
        self.m = len(r[0])
        self.lb = max(sum(self.q), sum([max(self.r[i]) for i in range(self.n)]))

    @classmethod
    def from_textio(cls, textio: TextIO) -> Problem:
        m, n, *q = map(int, textio.readline().split())
        r = [list(map(int, textio.readline().split())) for _ in range(n)]
        return cls(q, r)

    def __str__(self):
        s = [str(self.n), str(self.m)]  # Corrected order of n and m
        for i in range(self.n):
            s.append(" ".join(map(str, self.r[i])))
        return "\n".join(s)

# ...

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

        print("Updated Solution:")
        print(new_solution)
        print("Is Feasible:", new_solution.is_feasible())
        print("Objective Value:", new_solution.objective())
        print("Lower Bound:", new_solution.lower_bound())