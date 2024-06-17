import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CXGate, RXGate, UnitaryGate, SXGate, XGate, UGate
from qatg import QATG
from qatg import QATGFault

class CrosstalkFault(QATGFault):
    def __init__(self):
        # Specify the gate type and the qubits involved in the fault
        super(CrosstalkFault, self).__init__([XGate, XGate], [0,1], "gateType: U, qubits: 0-1, crosstalk affecting qubit 2")
        self.gates = [XGate, XGate]
    def createOriginalGate(self):
        # Create and return the original (fault-free) gate
        if len(self.gates) != 2:
            raise ValueError("The input array must contain exactly two gates.")
        gate1, gate2 = self.gates[0], self.gates[1]
        # identity_2 = np.eye(2)
        # identity_4 = np.eye(4)
        return UnitaryGate(np.kron(gate1().to_matrix(), gate2().to_matrix()), label=f'Unitary(origin)')

    def createFaultyGate(self, faultfreeGate):
        
        gate1, gate2 = self.gates[0], self.gates[1]

        # Initialize identity matrices for 2-qubit system
        identity_2 = np.eye(2)
        identity_4 = np.eye(4)

        # Define crosstalk effect as small RX rotations
        crosstalk_angle = 0.05 * np.pi
        crosstalk_gate = RXGate(crosstalk_angle)

        base_matrix_1 = np.kron(gate1().to_matrix(), identity_2)
        crosstalk_matrix_1 = np.kron(identity_2, crosstalk_gate.to_matrix())
        base_matrix_2 = np.kron(identity_2, gate2().to_matrix())
        crosstalk_matrix_2 = np.kron(crosstalk_gate.to_matrix(), identity_2)
        composite_matrix = np.matmul(crosstalk_matrix_1, base_matrix_1)
        composite_matrix = np.matmul(composite_matrix, crosstalk_matrix_2)
        composite_matrix = np.matmul(composite_matrix, base_matrix_2)
        return UnitaryGate(composite_matrix, label=f'Unitary(faulty)')

    def isSameGateType(self, gate):
        super().isSameGateType(gate)
		# print(len(gate), self.gateType)
        if len(self.gateType) == 1 or len(gate)==1:
            return isinstance(gate[0], self.gateType[0])
        else:
			# print(gate, self.gateType, isinstance(gate[0], self.gateType[0]), isinstance(gate[1], self.gateType[1]))
            return isinstance(gate[0], self.gateType[0]) and isinstance(gate[1], self.gateType[1]) or isinstance(gate[1], self.gateType[0]) and isinstance(gate[0], self.gateType[1])

# Example usage
# fault = CrosstalkFault([XGate, SXGate])
# original_gate = fault.createOriginalGate()
# print(fault.createOriginalGate(), fault.createFaultyGate(original_gate))
generator = QATG(circuitSize = 2, basisSingleQubitGateSet = [UGate], circuitInitializedStates = {2: [1, 0, 0, 0]}, minRequiredStateFidelity = 0.1)
configurationList = generator.createTestConfiguration([CrosstalkFault()])


for configuration in configurationList:
    print(configuration)
    # configuration.circuit.draw('mpl')
input()
