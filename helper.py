import numpy as np
from gates import single_gate_matrix,double_gate_matrix

def gates_and_identity(circuit):

    num_qubits = circuit.num_qubits
    data = circuit.data
    
    # Track which gates are applied to each qubit at each depth
    depth_gates = []
    current_depth_gates = ['id' for _ in range(num_qubits)]
    
    for instruction in data:
        gate_name = instruction.operation.name
        qubits = [circuit.qubits.index(q) for q in instruction.qubits]
        
        if len(qubits) == 2:
            # For two-qubit gates
            if any(current_depth_gates[q] != 'id' for q in qubits):
                depth_gates.append(current_depth_gates)
                current_depth_gates = ['id' for _ in range(num_qubits)]
            
            if qubits[0] > qubits[1]:
                temp = qubits[0]
                qubits[0] = qubits[1]
                qubits[1] = temp
                
            # Create new depth for the two-qubit gate
            new_depth = ['id' for _ in range(num_qubits)]
            new_depth[qubits[0]] = gate_name  # Put the gate name in the position of the first qubit
            new_depth = new_depth[:qubits[0]+1] + new_depth[qubits[1]+1:]  # Remove the 'id' after the gate
            depth_gates.append(new_depth)
            
        else:
            # For single-qubit gates
            if current_depth_gates[qubits[0]] != 'id':
                depth_gates.append(current_depth_gates)
                current_depth_gates = ['id' for _ in range(num_qubits)]
            current_depth_gates[qubits[0]] = gate_name
    
    # Add the last depth if there are any gates
    if any(g != 'id' for g in current_depth_gates):
        depth_gates.append(current_depth_gates)
    
    return depth_gates


# Create gate dictionary from the quantum circuit
def create_gate_dict(circuit):
    gate_dict = {"id": np.eye(2)}  # Ensure the identity gate is included
    
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        params = instruction.operation.params
        qubits = [circuit.qubits.index(q) for q in instruction.qubits]
        
        if gate_name not in gate_dict:
            if len(instruction.qubits) == 1:
                gate_dict[gate_name] = single_gate_matrix(gate_name, params)
            elif len(instruction.qubits) == 2:
                control, target = qubits
                gate_dict[gate_name] = double_gate_matrix(gate_name, control, target, params)
    
    return gate_dict


# Reverse gate order
def reverse_gate_order(depth_gates):
    return [gate_set[::-1] for gate_set in depth_gates]

# Tensor product of gates at each depth
def tensor_product_gates(depth_gates, gate_dict):
    num_qubits = len(depth_gates[0])
    result_matrices = []

    for gate_set in depth_gates:
        tensor_product = np.eye(1)  # Start with the identity for the entire system
        for gate in gate_set:
            tensor_product = np.kron(tensor_product, gate_dict[gate])
        result_matrices.append(tensor_product)

    return result_matrices

def statevector_simulator(circuit):
    num_qubits = circuit.num_qubits
    a = np.array([1, 0])

    # Initialize s_vector with the base vector 'a'
    s_vector = a

    # Apply the Kronecker product across all qubits
    for _ in range(1, num_qubits):
        s_vector = np.kron(s_vector, a)

    return s_vector