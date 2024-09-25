from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram


# Solves qubo on qpu. Returns list of solutions.
def solve_qubo(qubo, solver_type = 'cpu'):
    X = qubo.get_dict()

    model = QuadraticProgram("qubo")

    var_names = set()
    quadratic = {}

    for index in X:
        x, y = index
        var_names.add(x)
        var_names.add(y)

        quadratic[(str(x), str(y))] = X[index]

    for var_name in var_names:
        model.binary_var(name=str(var_name))

    model.minimize(quadratic=quadratic)
    
    # print(model.prettyprint())

    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    exact_mes = NumPyMinimumEigensolver()

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    exact = MinimumEigenOptimizer(exact_mes)

    exact_result = exact.solve(model)
    # print(exact_result.variables_dict)

    sample = {}

    for index in var_names:
        sample[index] = exact_result.variables_dict[str(index)]

    return sample
