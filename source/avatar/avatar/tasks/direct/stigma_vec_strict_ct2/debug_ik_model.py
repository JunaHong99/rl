
from ik_solver import FrankaIKSolver

solver = FrankaIKSolver()
print(f"Model nq (config dim): {solver.model.nq}")
print(f"Model nv (velocity dim): {solver.model.nv}")
print("Joint names:")
for name in solver.model.names:
    print(name)
