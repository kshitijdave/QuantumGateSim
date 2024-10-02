from qiskit import *
from gates import single_gate_matrix, double_gate_matrix, generalized_controlled_gate
from helper import gates_and_identity,create_gate_dict,reverse_gate_order,tensor_product_gates, statevector_simulator
import numpy as np
from qiskit_aer import Aer

# Define the quantum circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(1,0)
qc.cz(2,1)
qc.cy(1,2)


# Print the gates and reversed gates
gates_ = gates_and_identity(qc)
print("gates:", gates_)

reversed_gates_ = reverse_gate_order(gates_)
print("reversed_gates_:", reversed_gates_)

# Create the gate dictionary based on the quantum circuit
gate_dict = create_gate_dict(qc)

# Calculate the tensor product of gates at each depth
result_matrices = tensor_product_gates(reversed_gates_, gate_dict)

# Statevector
s_vector = statevector_simulator(qc)

for matrix in reversed(result_matrices[:]):
    s_vector = np.dot(matrix, s_vector)






backend = Aer.get_backend("statevector_simulator")

result = backend.run(qc, shots = 1024).result()

state_matrix = result.get_statevector()

print(state_matrix)



if np.allclose(s_vector, state_matrix):
    print("Both statevectors are the same.")
else:
    print("Both statevectors are different.")
    