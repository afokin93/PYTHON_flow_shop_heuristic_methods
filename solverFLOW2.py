from __future__ import annotations
from typing import TextIO, Optional, Any
import sys , random, operator, time

import Flow_v3 as flow


# -------------------------------------------------


# First improvement (local search)
def first_improvement(s):
    vi = s.local_moves() #VI movement iterator
    v = next(vi, None)
    while v is not None: #eqto houver movimentos locais
        incr = s.objective_incr_local(v) #melhor incremento
        #print(incr)
        if incr < 0:
            s.step(v)
            vi = s.local_moves()
        v = next(vi, None) 
    return s

# Best improvement (local search)
def best_improvement(s):
    vi = s.local_moves() #VI movement iterator
    v = next(vi, None)
    while v is not None: #eqto houver movimnetos locais
        best_v, best_incr = v, s.objective_incr_local(v) #melhor incremento
        print(best_incr)
        for v in vi:
            incr = s.objective_incr_local(v)
            if incr < best_incr: # problema de minimização, daí querer ue seja melhor
                best_v, best_incr = v, incr #atualiza
        if best_incr < 0:
            s.step(best_v)
            vi = s.local_moves()
        v = next(vi, None)
    return s

# Greedy construction (constructive method)
def greed_construction(problem):
    s = problem.empty_solution()
    ci = s.add_moves() #todas as componentes que podemos add
    c = next(ci, None) #considerar a 1a, se nõa houver usar none, solucao vazia
    while c is not None:
        best_c, best_incr = c, s.lower_bound_incr_add(c) #melhor incremento
#        print(best_incr)
        for c in ci:
            incr = s.lower_bound_incr_add(c)
            if incr < best_incr: # problema de minimização, daí querer ue seja melhor
                best_c, best_incr = c, incr #atualiza
        s.add(best_c)
        ci = s.add_moves()
        c = next(ci, None)
    return s

# Construção gulosa com desempate aleatório (meta)
def greed_construction_random_tie_breaking(problem):
    s = problem.empty_solution()
    ci = s.add_moves() #todas as componentes que podemos add
    c = next(ci, None) #considerar a 1a, se nõa houver usar none, solucao vazia
    while c is not None:
        best_c, best_incr = [c], s.lower_bound_incr_add(c) #melhor incremento
#        print(best_incr)
        for c in ci:
            incr = s.lower_bound_incr_add(c)
            if incr < best_incr: # problema de minimização, daí querer ue seja melhor
                best_c, best_incr = [c], incr #atualiza
            elif incr == best_incr: #caso empate, aleatorio
                best_c.append(c)
        s.add(random.choice(best_c))
        ci = s.add_moves()
        c = next(ci, None)
    return s

# Construção gulosa com aleatoreidade adaptativa (meta)
def greed_randomize_adaptive_construction(problem, alpha=0): #aqui vamos por numa lista para ir comparando os valores, mais flexibilidade, alfa pequeno gera valores melhores
    s = problem.empty_solution()
    cl = [(s.lower_bound_incr_add(c), c) for c in s.add_moves()] #candidate list e enumera elas
    while len(cl) != 0:
        c_min = min(cl, key=operator.itemgetter(0))[0]
        c_max = max(cl, key=operator.itemgetter(0))[0]
        thresh = c_min + alpha * (c_max - c_min)
        rcl = [c for incr, c in cl if incr <= thresh] # restrict candidate list, todas as componentes para
        s.add(random.choice(rcl))
        cl = [(s.lower_bound_incr_add(c), c) for c in s.add_moves()] # reinicia cl para o while continuar
    return s

# Construção meta GRASP
def grasp(problem, budget, alpha=0):
    start = time.perf_counter()
    best_s = greed_randomize_adaptive_construction(problem, alpha) #melhor solucao é a primeira
    best_obj = best_s.objective()
    print("GRASP ...")
    while time.perf_counter() - start < budget: #enquanto nao tiver passado tempo o suficiente
        s = greed_randomize_adaptive_construction(problem, alpha)
        obj = s.objective() #valor objetivo
        if obj < best_obj:
            best_s, best_obj = s, obj
            print(obj)
    return best_s


# -------------------------------------------------


if __name__ == '__main__':
    prob = flow.Problem.from_textio(sys.stdin) # Instanciar o problema
    ### PARA PROCURA LOCAL
    sol1 = first_improvement(prob.initial_solution())
    print("FIRST IMPRV | Obj:", "{} --> ".format(sol1.objective()), 'Sol:', sol1)
    sol2 = best_improvement(prob.initial_solution())
    print("BEST IMPRV  | Obj:", "{} --> ".format(sol2.objective()), 'Sol:', sol2)
    ### PARA MÉTODOS CONSTRUTIVOS E MISTOS
    sol3 = greed_construction(prob)
    print("GREEDY      | Obj:", "{} --> ".format(sol3.objective()), 'Sol:', sol3)
    sol4 = greed_construction_random_tie_breaking(prob)
    print("GREEDY RAND | Obj:", "{} --> ".format(sol4.objective()), 'Sol:', sol4)
    sol5 = greed_randomize_adaptive_construction(prob, 0.01) #alfa pequeno melhor
    print("GREEDY ADPT | Obj:", "{} --> ".format(sol5.objective()), 'Sol:', sol5)
    sol6 = grasp(prob, 2, 0.01)
    print("GRASP       | Obj:", "{} --> ".format(sol6.objective()), 'Sol:', sol6)