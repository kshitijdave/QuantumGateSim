import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import numpy as np
from qiskit_aer import Aer
from gates import single_gate_matrix, double_gate_matrix, generalized_controlled_gate
from helper import gates_and_identity, create_gate_dict, reverse_gate_order, tensor_product_gates, statevector_simulator

# Function to create a complex quantum circuit with rotational gates and controlled-X gates
def create_complex_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)  # Apply Hadamard gate to all qubits
        qc.rx(np.pi / 4, i)  # Add rotation around X-axis
        qc.ry(np.pi / 6, i)  # Add rotation around Y-axis
        qc.rz(np.pi / 8, i)  # Add rotation around Z-axis
    for i in range(0, n_qubits - 1):
        qc.cx(i, i + 1)  # Controlled-X gate
    return qc


# Function to run the custom statevector simulator
def run_custom_statevector_simulator(qc):
    gates_ = gates_and_identity(qc)
    reversed_gates_ = reverse_gate_order(gates_)
    gate_dict = create_gate_dict(qc)
    result_matrices = tensor_product_gates(reversed_gates_, gate_dict)
    
    s_vector = statevector_simulator(qc)
    for matrix in reversed(result_matrices[:]):
        s_vector = np.dot(matrix, s_vector)
    return s_vector


# Function to run the custom unitary simulator
def run_custom_unitary_simulator(qc):
    gates_ = gates_and_identity(qc)
    reversed_gates_ = reverse_gate_order(gates_)
    gate_dict = create_gate_dict(qc)
    result_matrices = tensor_product_gates(reversed_gates_, gate_dict)
    
    final = result_matrices[-1]
    for matrix in reversed(result_matrices[:-1]):
        final = np.dot(final, matrix)
    return final


def statevector_simulator(qc):
    n_qubits = qc.num_qubits
    # Initialize the statevector to |0>^n
    statevector = np.zeros(2**n_qubits, dtype=complex)
    statevector[0] = 1.0
    return statevector


# Function to run Qiskit's simulators
def run_qiskit_backend(qc, backend_name):
    backend = Aer.get_backend(backend_name)
    result = backend.run(qc).result()
    if backend_name == "statevector_simulator":
        return result.get_statevector()
    elif backend_name == "unitary_simulator":
        return result.get_unitary()


def compare_simulators():
    # Circuit sizes to compare (up to 13 qubits)
    circuit_sizes = range(5, 13)  # Number of qubits from 5 to 13
    custom_statevector_times = []
    qiskit_statevector_times = []
    custom_unitary_times = []
    qiskit_unitary_times = []

    # Measure time for each circuit size
    for n_qubits in circuit_sizes:

        print(f"\nAnalyzing for {n_qubits} qubit ")
        qc = create_complex_circuit(n_qubits)

        # Custom Statevector Simulator
        start = time.time()
        _ = run_custom_statevector_simulator(qc)
        custom_statevector_times.append(time.time() - start)

        # Qiskit's Statevector Simulator
        start = time.time()
        _ = run_qiskit_backend(qc, "statevector_simulator")
        qiskit_statevector_times.append(time.time() - start)

        # Custom Unitary Simulator
        start = time.time()
        _ = run_custom_unitary_simulator(qc)
        custom_unitary_times.append(time.time() - start)

        # Qiskit's Unitary Simulator
        start = time.time()
        _ = run_qiskit_backend(qc, "unitary_simulator")
        qiskit_unitary_times.append(time.time() - start)

    # Plot Statevector Results
    plt.figure(figsize=(12, 6))
    plt.plot(circuit_sizes, custom_statevector_times, marker='o', label='Custom Statevector Simulator')
    plt.plot(circuit_sizes, qiskit_statevector_times, marker='o', label="Qiskit's Statevector Simulator")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Statevector Simulation Execution Times")
    plt.legend()
    plt.grid()
    plt.savefig("statevector_simulation_times.png")  # Save the figure
    plt.show()

    # Plot Unitary Results
    plt.figure(figsize=(12, 6))
    plt.plot(circuit_sizes, custom_unitary_times, marker='s', label='Custom Unitary Simulator')
    plt.plot(circuit_sizes, qiskit_unitary_times, marker='s', label="Qiskit's Unitary Simulator")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Unitary Simulation Execution Times")
    plt.legend()
    plt.grid()
    plt.savefig("unitary_simulation_times.png")  # Save the figure
    plt.show()


if __name__ == "__main__":
    compare_simulators()
