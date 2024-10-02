import numpy as np
import qiskit.circuit.library.standard_gates as std_gates
from typing import Optional, List
from qiskit.exceptions import QiskitError

########## FUNCTIONS ##############



# Define single and double gate matrices
def single_gate_matrix(gate: str, params: Optional[List[float]] = None):
    """Returns the matrix representation of a single gate."""
    if params is None:
        params = []

    if gate == "U":
        gc = std_gates.UGate
    elif gate == "u3":
        gc = std_gates.U3Gate
    elif gate == "h":
        gc = std_gates.HGate
    elif gate == "u":
        gc = std_gates.UGate
    elif gate == "p":
        gc = std_gates.PhaseGate
    elif gate == "u2":
        gc = std_gates.U2Gate
    elif gate == "u1":
        gc = std_gates.U1Gate
    elif gate == "rz":
        gc = std_gates.RZGate
    elif gate == "id":
        return np.eye(2)
    elif gate == "sx":
        gc = std_gates.SXGate
    elif gate == "x":
        gc = std_gates.XGate
    elif gate == "y":
        gc = std_gates.YGate
    elif gate == "z":
        gc = std_gates.ZGate
    elif gate == "rx":
        gc = std_gates.RXGate
    elif gate == "ry":
        gc = std_gates.RYGate
    elif gate == "rz":
        gc = std_gates.RZGate
    else:
        raise QiskitError("Gate is not a valid basis gate for this simulator: %s" % gate)
    
    return gc(*params).to_matrix()


# These simulators support gates involving up to 2 qubits, including CX, CY, and CZ gates.

def generalized_controlled_gate(gate: str, control: int, target: int):
    
    # Define the identity, X, Y, and Z matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Define the outer products for the control qubit states
    a = np.array([1, 0])  # state |0>
    b = np.array([0, 1])  # state |1>

    # Construct the |0><0| term for the control qubit
    P0 = np.outer(a, a)

    # Construct the |1><1| term for the control qubit
    P1 = np.outer(b, b)

    # Select the gate matrix for the target qubit
    if gate == "cx":
        target_gate = X
    elif gate == "cy":
        target_gate = Y
    elif gate == "cz":
        target_gate = Z
    else:
        raise ValueError(f"Gate '{gate}' is not supported.")
    

    # Initialize the Kronecker product terms
    first_term = 1
    second_term = 1

    res_f = 1
    res_s = 1
    # Handle the case where control < target
    if control < target:
        for _ in range(1, target - control):
            res_f = np.kron(res_f, I)
            res_s = np.kron(res_s, I)
        res_f = np.kron(res_f, P0)
        res_s = np.kron(res_s, P1)

        first_term = np.kron(I, res_f)
        second_term = np.kron(target_gate, res_s)
    # The CNOT gate is the sum of the two terms

    if target < control:
        for _ in range(1, control - target):
            res_f = np.kron(res_f, I)
            res_s = np.kron(res_s, I)
        res_f = np.kron(res_f, I)
        res_s = np.kron(res_s, target_gate)

        first_term = np.kron(P0, res_f)
        second_term = np.kron(P1, res_s)
    
    controlled_gate = first_term + second_term
    return controlled_gate

def double_gate_matrix(gate: str, control: int, target: int, params: Optional[List[float]] = None):
    """ Returns the matrix representation of a double gate with given control and target qubits."""
    return generalized_controlled_gate(gate, control, target)


