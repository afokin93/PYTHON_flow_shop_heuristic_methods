from __future__ import annotations
from typing import TextIO, Optional, Any
import sys , random, operator, time

import FLOWV1 as flow

def greed_construction(problem):
    s = problem.empty_solution()
    ci = s.add_moves() #todas as componentes que podemos add
    c = next(ci, None) #considerar a 1a, se nõa houver usar none, solucao vazia
    while c is not None:
        best_c, best_incr = c, s.lower_bound_incr_add(c) #melhor incremento
        for c in ci:
            incr = s.lower_bound_incr_add(c)
            if incr < best_incr: # problema de minimização, daí querer ue seja melhor
                best_c, best_incr = c, incr #atualiza
        s.add(best_c)
        ci = s.add_moves()
        c = next(ci, None)
    return s


def best_improvement(s):
    vi = s.local_moves() #VI movement iterator
    v = next(vi, None)
    while v is not None: #eqto houver movimnetos locais
        best_v, best_incr = v, s.objective_incr_local(v) #melhor incremento
        for v in vi:
            incr = s.objective_incr_local(v)
            if incr < best_incr: # problema de minimização, daí querer ue seja melhor
                best_v, best_incr = v, incr #atualiza
        if best_incr < 0:
            s.step(best_v)
            vi = s.local_moves()
        v = next(vi, None)
    return s


def first_improvement(s):
    vi = s.local_moves() #VI movement iterator
    v = next(vi, None)
    while v is not None: #eqto houver movimnetos locais
        incr = s.objective_incr_local(v) #melhor incremento
        if incr < 0:
            s.step(v)
            vi = s.local_moves()
        v = next(vi, None) 
    return s

def greed_construction_random_tie_breaking(problem):
    s = problem.empty_solution()
    ci = s.add_moves() #todas as componentes que podemos add
    c = next(ci, None) #considerar a 1a, se nõa houver usar none, solucao vazia
    while c is not None: 
        best_c, best_incr = [c], s.lower_bound_incr_add(c) #melhor incremento
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

def greed_randomize_adaptive_construction(problem, alpha=0): #aqui vamos por numa lista para ir comparando os valores, mais flexibilidade, alfa pequeno gera valores melhores
    s = problem.empty_solution()     
    cl = [(s.lower_bound_incr_add(c), c) for c in s.add_moves()] #candidate list e enumera elas
    while len(cl) != 0:
        c_min = min(cl, key=operator.itemgetter(0))[0]
        c_max = max(cl, key=operator.itemgetter(0))[0]
        thresh = c_min + alpha * (c_max - c_min)
        rcl = [c for incr, c in cl if incr <= thresh] #restrict candidate list, todas as componentes para
        s.add(random.choice(rcl))
        cl = [(s.lower_bound_incr_add(c), c) for c in s.add_moves()] #reinicia cl para o while continuar
    return s

def grasp(problem, budget, alpha=0):
    start = time.perf_counter()
    best_s = greed_randomize_adaptive_construction(problem, alpha) #melhor solucao é a primeira
    best_obj = best_s.objective()
    while time.perf_counter() - start < budget: #enquanto nao tiver passado tempo o sudifiente
        s = greed_randomize_adaptive_construction(problem, alpha)
        obj = s.objective() #valor objetivo
        if obj < best_obj:
            best_s, best_obj = s, obj
            print(obj)
    return best_s

if __name__ == '__main__':
    prob = flow.Problem.from_textio(sys.stdin) #instanciar o problema
    
    #sol1 = greed_construction(prob)
    #print("Obj:", "{:.2f}".format(sol1.objective()), 'Sol:', sol1)
    #sol2 = greed_construction_random_tie_breaking(prob)
    #print("Obj:", "{:.2f}".format(sol2.objective()), 'Sol:', sol2)
    #sol3 = greed_randomize_adaptive_construction(prob, 0.01) #alfa pequeno melhor
    #print("Obj:", "{:.2f}".format(sol3.objective()), 'Sol:', sol3)
    #sol4 = grasp(prob, 2, 0.01)
    #print("Obj:", "{:.2f}".format(sol4.objective()), 'Sol:', sol4)
    #sol5 = best_improvement(prob.empty_solution())
    #print("Obj:", "{:.2f}".format(sol5.objective()), 'Sol:', sol5)
    #sol6 = first_improvement(prob.empty_solution())
    #print("Obj:", "{:.2f}".format(sol6.objective()), 'Sol:', sol6)
    #for i in range(10):
        #sol1 = greed_construction(prob)
        #print(f'Run {i+1} - Obj: {"{:.2f}".format(sol1.objective())}, Sol: {sol1}')

    # Run greed_construction_random_tie_breaking 10 times
    start_time1 = time.time()
    for i in range(10):
        start_time = time.time()
        sol2 = greed_construction_random_tie_breaking(prob)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Run {i+1} - Obj: {"{:.2f}".format(sol2.objective())}, Sol: {sol2}, Time: {execution_time}')
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    print(f'Mean Time : {execution_time1/10}')
    # Run greed_randomize_adaptive_construction 10 times
    
    start_time2 = time.time()
    for i in range(10):
        start_time = time.time()
        sol3 = greed_randomize_adaptive_construction(prob, 0.01)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Run {i+1} - Obj: {"{:.2f}".format(sol3.objective())}, Sol: {sol3}, Time: {execution_time}')
    end_time2 = time.time()
    execution_time2 = end_time2 - start_time2
    print(f'Mean Time : {execution_time2/10}')
    # Run grasp 10 times
    start_time3 = time.time()
    for i in range(10):
        start_time = time.time()
        sol4 = grasp(prob, 2, 0.01)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Run {i+1} - Obj: {"{:.2f}".format(sol4.objective())}, Sol: {sol4}, Time: {execution_time}')
    end_time3 = time.time()
    execution_time3 = end_time3 - start_time3
    print(f'Mean Time : {execution_time3/10}')