from __future__ import annotations
from typing import TextIO, Optional, Any
import sys, random


# ---------------


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
    def __init__(self, problem, used, unused, t, ms, lb): # inicializa a solução
        # init serve para embrulhar as variáveis
        self.problem = problem
        self.used = used # trabalho usado
        self.unused = unused # trabalho não usado
        self.t = t # tempo de processamento em cada máquina
        self.ms = ms # makespan
        self.lb = lb # lower bound
    
    def __str__(self):
        return " ".join([str(u) for u in self.used])
    
    def copy(self) -> Solution:
        return self.__class__(self.problem,
                              self.used.copy(),
                              self.unused.copy(),
                              self.t.copy(),
                              self.ms,
                              self.lb)
    
    def is_feasible(self) -> bool:
        return True # sempre # Return whether the solution is feasible or not
    
    def objective(self) -> Optional[Objective]:
        return self.ms# Return the objective value for this solution if defined, otherwise should return None
    
    def lower_bound(self) -> Optional[Objective]:
        return self.lb # Return the lower bound value for this solution if defined, otherwise return None
    
    def _can_add(self, jobid): ### PARA MÉTODO CONSTRUTIVO
        return True # adicionar função que diga que só se pode adicionar quando estiver acabado no anterior
    
    def _can_swap(self, jobid_add, jobid_rem): ### PARA PROCURA LOCAL
        return True # sempre se pode trocar dois trabalhos
    
    def add_moves(self) -> Iterable[Component]: ### PARA MÉTODO CONSTRUTIVO
        # mostra todos os componentes que podem ser adicionados à solução
        for jobid in self.unused:
            if self._can_add(jobid):
                yield Component(jobid) # yield: objeto que guarda resultados de funções
    
    '''def local_moves(self) -> Iterable[LocalMove]:
       # primeira função de movimentos DE PROCURA LOCAL --> 1670
        used_list = list(self.used)
        for i in range(len(used_list)):
            for j in range(i + 1, len(used_list)):
                jobid_add, jobid_rem = used_list[i], used_list[j]
                if self._can_swap(jobid_add, jobid_rem):
                    yield LocalMove(jobid_add, jobid_rem)'''

    '''def generate_swap_moves(self):
        # segunda função de movimentos DE PROCURA LOCAL --> 1637
        swap_moves = []
        for i in range(len(self.used) - 1):
            move_pairs = (self.used[i], self.used[i + 1])
            swap_moves.append(move_pairs)
        return swap_moves'''
    '''def local_moves(self):
        swap_moves = self.generate_swap_moves()
        for move_pairs in swap_moves:
            jobid_add, jobid_rem = move_pairs
            if self._can_swap(jobid_add, jobid_rem):
                yield LocalMove(jobid_add, jobid_rem)'''

    def generate_swap_moves(self, start_sequence):
    # Terceira função de movimentos DE PROCURA LOCAL --> 1514
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
            jobid_add, jobid_rem = move_pairs
            if self._can_swap(jobid_add, jobid_rem):
                new_used = self.step(LocalMove(jobid_add, jobid_rem))
                if new_used is not None and tuple(new_used) not in seen_sequences:
                    seen_sequences.add(tuple(new_used))
                    yield LocalMove(jobid_add, jobid_rem)

    def _add_job(self, jobid): ### PARA MÉTODO CONSTRUTIVO
    # função auxiliar para ajudar na próxima função / adiciona um trabalho à solução
        prob = self.problem
        self.used.append(jobid)
        self.unused.remove(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.t[i] += prob.r[i][jobid] # o tempo de processamento é igual ao tempo do trabalho
            else: # se não for a primeira máquina
                self.t[i] = max(self.t[i], self.t[i-1]) + prob.r[i][jobid] # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, mais o tempo do trabalho
        self.ms= self.t[-1] # o makespan é o tempo da última máquina
    
    def _remove_job(self, jobid): ### PARA MÉTODO CONSTRUTIVO
    # função auxiliar para ajudar na próxima função / remove um trabalho à solução
        prob = self.problem
        self.used.remove(jobid)
        self.unused.add(jobid)
        for i in range(prob.n): # para cada máquina
            if i == 0: # se for a primeira máquina
                self.t[i] -= prob.r[i][jobid] # o tempo de processamento é reduzido pelo tempo do trabalho
            else: # se não for a primeira máquina
                self.t[i] = max(self.t[i] - prob.r[i][jobid], self.t[i-1]) # o tempo de processamento é o máximo entre o tempo da máquina anterior e o tempo da máquina atual, menos o tempo do trabalho
        self.ms= self.t[-1] # o makespan é o tempo da última máquina
    
    def add(self, c: Component) -> None: ### PARA MÉTODO CONSTRUTIVO
    # adiciona um componente à solução
        self._add_job(c.jobid)
        self.lb = self._lb_update_add(c.jobid) # atualiza o lower bound

    def step(self, lmove): ### PARA PROCURA LOCAL
    # Função auxiliar para ajudar na próxima função / troca dois jobs na solução completa
        prob = self.problem
        # Extrai os trabalhos da LocalMove
        jobid_add = lmove.jobid_add
        jobid_rem = lmove.jobid_rem
        # Create a copy of the set before modification
        new_used = list(jobid_rem if jobid == jobid_add else (jobid_add if jobid == jobid_rem else jobid) for jobid in self.used)
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
        lb = self._lb_update_local(jobid) # calcula o novo lower boundos
    
    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]: ### PARA PROCURA LOCAL
        incr = (self.problem.r[-1][lmove.jobid_rem] if lmove.jobid_rem is not None else 0)\
             - (self.problem.r[-1][lmove.jobid_add] if lmove.jobid_add is not None else 0) # o incremento da função objetivo é a diferença entre os tempos dos trabalhos na última máquina
        return incr
    
    def objective_incr_add(self, c: Component) -> Optional[Objective]: ### PARA MÉTODO CONSTRUTIVO
        """Return the objective value increment resulting from adding a component.
        If the objective value is not defined after adding the component return None."""
        return self.problem.r[-1][c.jobid] # o incremento da função objetivo é o tempo do trabalho na última máquina
    
    def lower_bound_incr_add(self, c: Component) -> Optional[Objective]: ### PARA MÉTODO CONSTRUTIVO
        """Return the lower bound increment resulting from adding a component.
        If the lower bound is not defined after adding the component, return None."""
        lb = self._lb_update_add(c.jobid) # calcula o novo lower bound
        return lb - self.lb # retorna a diferença entre o novo e o antigo lower bound
    
    #L1
    '''def _lb_update_add(self, jobid):
        prob = self.problem
        machine_loads = [0] * prob.m  # Lista para armazenar as cargas de cada máquina
        for i in range(prob.n):  # Para cada trabalho na solução
            for j in range(prob.m):  # Para cada máquina
                if i in self.used:
                    machine_loads[j] += prob.r[i][j]  # Adiciona o tempo de processamento do trabalho na máquina
        return max(machine_loads)  # Retorna a maior carga entre todas as máquinas'''
    #L2
    '''def _lb_update_add(self, jobid):
        prob = self.problem
        machine_loads = [0] * prob.m  # Lista para armazenar as cargas de cada máquina[^3^][3]
        for i in range(prob.n):  # Para cada trabalho na solução
            for j in range(prob.m):  # Para cada máquina[^4^][4][^5^][5]
                if i in self.used:
                    machine_loads[j] += prob.r[i][j]  # Adiciona o tempo de processamento do trabalho na máquina[^6^][6]
        # Calculate P_i for each machine
        P = []
        for i in range(prob.m):
            if i < len(prob.r):  # Check if i is within the range of indices for prob.r
                P_i = sum(prob.r[i])  # Sum of processing times on machine i
                if i > 0:
                    P_i += min([sum(prob.r[r][:i]) for r in range(prob.n)])  # Add min of sums on preceding machines
                if i < prob.m - 1:
                    P_i += min([sum(prob.r[r][i+1:]) for r in range(prob.n)])  # Add min of sums on subsequent machines
                P.append(P_i)
        # Set L^+_M to the maximum of P_i values
        L_plus_M = max(P)
        return L_plus_M'''
    ''' 
    #L3 
    def _lb_update_add(self, jobid):
        prob = self.problem
        job_loads = [0] * prob.n  # List to store the loads of each job
        for i in range(prob.n):  # For each job in the solution
            for j in range(prob.m):  # For each machine
                if i in self.used:
                    job_loads[i] += prob.r[i][j]  # Add the processing time of the job on the machine

        # Calculate L_J
        L_J = max(job_loads)
        return L_J
    '''
    #L4
    '''def _lb_update_add(self, jobid):
        prob = self.problem
        T = [0] * prob.n  # List to store the Q_j values for each job
        for j in range(prob.n):  # For each job in the solution[^3^][3]
            T[j] = sum(prob.r[j])  # Sum of processing times on job j
            for s in range(prob.n):  # For each other job
                if s != j:
                    T[j] += min(prob.r[s][0], prob.r[s][-1])  # Add min of first and last processing times
        # Calculate L'_J
        L_prime_J = max(Q)
        return L_prime_J'''
    #L5
    def _lb_update_add(self, jobid): ### PARA MÉTODO CONSTRUTIVO
        prob = self.problem
        # Step 1: For each machine i, sort the p_{ij} values in non-decreasing order.
        if prob.m <= len(prob.r):
            tau = [sorted(prob.r[i]) for i in range(prob.m)]
        else:
            #print("Error: The number of machines is greater than the length of prob.r.")
            return 0
        # Step 2: Let sigma_{ik} denote sum_{k'=1}^{k} tau_{i,k'}.
        sigma = [[sum(tau[i][:k+1]) for k in range(prob.n)] for i in range(prob.m)]
        # Step 3: Compute a lower bound gamma_{ik} on the time at which machine i finishes processing the kth job in the sequence.
        gamma = [[0]*prob.n for _ in range(prob.m)]
        # Step 4: For all k, gamma_{1k} is set to sigma_{1k}.
        gamma[0] = sigma[0]
        # Step 5: For all i, gamma_{i1} is set to min_{j} {sum_{i'=1}^{i} p_{i',j}}.
        for i in range(1, prob.m):
            gamma[i][0] = min([sum(prob.r[i_prime][j] for i_prime in range(i+1)) for j in range(prob.n)])
        # Step 6: For i = 2, ..., m and k = 2, ..., n, gamma_{ik} is set to the larger of the following four values:
        for i in range(1, prob.m):
            for k in range(1, prob.n):
                beta_1ik = sigma[i][k] + gamma[i-1][0]
                beta_2ik = sigma[i][k-1] + gamma[i][0]
                beta_3ik = max([gamma[i_prime][k-1] + min([sum(prob.r[i_double_prime][j] for i_double_prime in range(i_prime, i+1)) for j in range(prob.n)]) for i_prime in range(i+1)])
                beta_4ik = max([gamma[i_prime][k] + min([sum(prob.r[i_double_prime][j] for i_double_prime in range(i_prime+1, i+1)) for j in range(prob.n)]) for i_prime in range(i)])
                gamma[i][k] = max(beta_1ik, beta_2ik, beta_3ik, beta_4ik)
        # Step 7: At the end of the procedure, gamma_{mn} is a lower bound for the PFM.
        return gamma[-1][-1]

    def _lb_update_local(self, jobid): ### PARA PROCURA LOCAL
    # Retorna atualização do lower bound [se local search]
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
            self.lb = max(sum(q), sum([max(r[i]) for i in range(self.n)]))

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
    
    def empty_solution(self) -> Solution: ### PARA MÉTODO CONSTRUTIVO
        return Solution(self, [], set(range(self.m)), [0] * self.n, 0, self.lb)

    def initial_solution(self) -> Solution: ### PARA PROCURA LOCAL
        return Solution(self, list(range(self.m)), [], self.t, self.lb, self.lb)


# ---------------


if __name__ == '__main__':
    # Read the problem from stdin
    problem = Problem.from_textio(sys.stdin)
    # Test your functions here
    print("Problem:")
    print(problem)

    ### PARA MÉTODOS CONSTRUTIVOS
    solution_c = problem.empty_solution() #Example of creating a solution and testing some methods
    print("\nEmpty solution - Initial for Construction:")
    print(solution_c.ms)
    print("\n")
    for move in solution_c.add_moves():
        #print("\nAdding job:", move)
        # Create a copy of the solution to keep the state across iterations
        new_solution_c = solution_c.copy()
        new_solution_c.add(move)
        # Print das demais informações
        #print("Updated Solution:")
        print(new_solution_c)
        #print("Is Feasible:", new_solution1.is_feasible())
        #print("Objective Value:", new_solution1.objective())
        #print("Lower Bound:", new_solution1.lower_bound())

    ### PARA PROCURA LOCAL
    solution_ls = problem.initial_solution() #Example of creating a solution and testing some methods
    print("\nSolution Zero - Initial for Local Search:")
    print(solution_ls.ms)
    print("\n")
    for move in solution_ls.local_moves():
        #print("\nAdding job:", move)
        # Create a copy of the solution to keep the state across iterations
        new_solution_ls = solution_ls.copy()
        new_solution_ls.step(move)
        # Print das demais informações
        #print("Updated Solution:")
        print(new_solution_ls)
        #print(move)
        #print("Is Feasible:", new_solution.is_feasible())
        #print("Objective Value:", new_solution.objective())
        #print("Lower Bound:", new_solution.lower_bound())
        incr = new_solution_ls.objective_incr_local(move)
        print("Increment:", incr)