

#### IMPORTS #####
from qiskit import QuantumCircuit
import numpy as np
import qiskit.circuit.library.standard_gates as std_gates

from qiskit_aer import Aer
from qiskit.visualization import array_to_latex
from gates import single_gate_matrix, double_gate_matrix, generalized_controlled_gate
from helper import gates_and_identity,create_gate_dict,reverse_gate_order,tensor_product_gates

# Define the quantum circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(1,0)
qc.cz(2,1)
qc.cy(1,2)

# Print the gates and reversed gates
gates_ = gates_and_identity(qc)

reversed_gates_ = reverse_gate_order(gates_)

# Create the gate dictionary based on the quantum circuit
gate_dict = create_gate_dict(qc)

# Calculate the tensor product of gates at each depth
result_matrices = tensor_product_gates(reversed_gates_, gate_dict)

# Start with the last matrix
final = result_matrices[-1]

for matrix in reversed(result_matrices[:-1]):
    final = np.dot(final, matrix)




# Compare the result with original


backend = Aer.get_backend("unitary_simulator")

result = backend.run(qc, shots = 1024).result()

unitary_matrix = result.get_unitary()

print("Unitary Matrix:")

print(unitary_matrix)


if np.allclose(final, unitary_matrix):
    print("Both matrices are the same.")
else:
    print("The matrices are different.")