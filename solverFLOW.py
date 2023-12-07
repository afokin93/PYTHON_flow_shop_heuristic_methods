from __future__ import annotations
from typing import TextIO, Optional, Any
import sys

import Flow as flow

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


if __name__ == '__main__':
    prob = flow.Problem.from_textio(sys.stdin) #instanciar o problema
    sol1 = greed_construction(prob)
    print("Obj:", "{:.2f}".format(sol1.objective()), 'Sol:', sol1)
    #sol2 = greed_construction_random_tie_breaking(prob)
    #print("Obj:", "{:.2f}".format(sol2.objective()), 'Sol:', sol2)
    #sol3 = greed_randomize_adaptive_construction(prob, 0.01) #alfa pequeno melhor
    #print("Obj:", "{:.2f}".format(sol3.objective()), 'Sol:', sol3)
    #sol4 = grasp(prob, 2, 0.01)
    #print("Obj:", "{:.2f}".format(sol4.objective()), 'Sol:', sol4)
    sol5 = best_improvement(prob.empty_solution())
    print("Obj:", "{:.2f}".format(sol5.objective()), 'Sol:', sol1.copy())
    sol6 = first_improvement(prob.empty_solution())
    print("Obj:", "{:.2f}".format(sol6.objective()), 'Sol:', sol1.copy())