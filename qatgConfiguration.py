import random
import numpy as np
from math import ceil
from scipy.stats import chi2, ncx2
# from qiskit.execute_function import execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import standard_errors, ReadoutError
# import sutff
import sys
import os.path as osp
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from crosstalk_scheduler import CrosstalkAdaptiveSchedule
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.models import BackendProperties
sys.path.append(osp.dirname(osp.abspath(__file__)))
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.scheduling import DynamicCircuitInstructionDurations
from qatg.fuck import get_simultaneuos_gates
from qatgUtil import *
from qiskit.circuit import Delay


random.seed(114514)

class QATGConfiguration():
	"""the return results of qatg is described as qatgConfiguration objects"""
	def __init__(self, circuitSetup: dict, simulationSetup: dict, faultObject):
		# circuitSetup: circuitSize, basisGateSet, quantumRegisterName, classicalRegisterName, circuitInitializedStates
		
		self.circuitSize = circuitSetup['circuitSize']
		self.basisGateSet = circuitSetup['basisGateSet']
		self.basisGateSetString = circuitSetup['basisGateSetString']
		self.circuitInitializedStates = circuitSetup['circuitInitializedStates']
		# self.backend = AerSimulator()
		self.backend = FakeJakartaV2()
		# service = QiskitRuntimeService(instance="ibm-q/open/main")
		# self.backend = service.backend('ibm_osaka')
		self.oneQubitErrorProb = simulationSetup['oneQubitErrorProb']
		self.twoQubitErrorProb = simulationSetup['twoQubitErrorProb']
		self.zeroReadoutErrorProb = simulationSetup['zeroReadoutErrorProb']
		self.oneReadoutErrorProb = simulationSetup['oneReadoutErrorProb']

		self.targetAlpha = simulationSetup['targetAlpha']
		self.targetBeta = simulationSetup['targetBeta']

		self.simulationShots = simulationSetup['simulationShots']
		self.testSampleTime = simulationSetup['testSampleTime']

		self.faultObject = faultObject

		quantumRegisterName = circuitSetup['quantumRegisterName']
		classicalRegisterName = circuitSetup['classicalRegisterName']
		self.quantumRegister = QuantumRegister(self.circuitSize, quantumRegisterName)
		self.classicalRegister = ClassicalRegister(self.circuitSize, classicalRegisterName)

		self.faultfreeQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)

		self.faultfreeDistribution = []
		self.faultyDistribution = []
		self.repetition = np.nan
		self.boundary = np.nan
		self.simulatedOverkill = np.nan
		self.simulatedTestescape = np.nan
		self.cktDepth = np.nan
		self.OnestateFidelity = np.nan

		self.noiseModel = self.getNoiseModel()

	def __str__(self):
		rt = ""
		rt += "Target fault: { " + str(self.faultObject) + " }\n"
		rt += "Length: " + str(self.cktDepth)
		rt += "\tRepetition: " + str(self.repetition)
		rt += "\tCost: " + str(self.cktDepth * self.repetition) + "\n"
		rt += "Chi-Value boundary: " + str(self.boundary) + "\n"
		rt += "State Fidelity: " + str(self.OnestateFidelity) + "\n"
		rt += "Overkill: "+ str(self.simulatedOverkill)
		rt += "\tTest Escape: " + str(self.simulatedTestescape) + "\n"
		# rt += "Circuit: \n" + str(self.faultfreeQCKT)

		return rt

	def getNoiseModel(self):
		# Depolarizing quantum errors
		oneQubitError = standard_errors.depolarizing_error(self.oneQubitErrorProb, 1)
		twoQubitError = standard_errors.depolarizing_error(self.twoQubitErrorProb, 2)
		qubitReadoutError = ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])

		# Add errors to noise model
		noiseModel = NoiseModel()
		noiseModel.add_all_qubit_quantum_error(oneQubitError, self.basisGateSetString)
		noiseModel.add_all_qubit_quantum_error(twoQubitError, ['cx'])
		noiseModel.add_all_qubit_readout_error(qubitReadoutError)

		return noiseModel

	def setTemplate(self, template, OnestateFidelity):
		# template itself is faultfree
		self.OnestateFidelity = OnestateFidelity

		qbIndexes = self.faultObject.getQubits()
		# print(template)
		# print("===========================")
		
		for gates in template:
			# in template, a list for seperate qubits and a gate for all qubits
			if isinstance(gates, list):
				print("list", gates)
				for k in range(len(gates)):
					self.faultfreeQCKT.append(gates[k], [qbIndexes[k]])
					if self.faultObject.isSameGateType([gates[k]]):
						# print("dnwinwindiwnidnwindiwdiwndiwnidniw")
						# print("---------------------------")
						self.faultyQCKT.append(self.faultObject.createFaultyGate(gates[k]), [qbIndexes[k]])
					else:
						self.faultyQCKT.append(gates[k], [qbIndexes[k]])
			elif issubclass(type(gates), Gate):
				# print("type", gates)
				# print("---------------------------")
				self.faultfreeQCKT.append(gates, qbIndexes)
				if self.faultObject.isSameGateType([gates]):
					self.faultyQCKT.append(self.faultObject.createFaultyGate(gates), qbIndexes)
				else:
					self.faultyQCKT.append(gates, qbIndexes)
			else:
				raise TypeError(f"Unknown object \"{gates}\" in template")

			self.faultfreeQCKT.append(qGate.Barrier(len(qbIndexes)), qbIndexes)
			self.faultyQCKT.append(qGate.Barrier(len(qbIndexes)), qbIndexes)
		self.faultfreeQCKT.measure(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT.measure(self.quantumRegister, self.classicalRegister)

		self.cktDepth = len(template)

		return

	def create_two_qubit_unitary(self,gate1, gate2):
		name1, params1, qubit1, _ = gate1
		name2, params2, qubit2, _ = gate2

		gate1_obj = qGate.SXGate() if name1 == 'sx' else None
		gate2_obj = qGate.SXGate() if name2 == 'sx' else None
		
		if gate1_obj and gate2_obj:  # Ensure both gates are non-RZ gates
			gate1_matrix = gate1_obj.to_matrix()
			gate2_matrix = gate2_obj.to_matrix()
			combined_matrix = np.kron(gate1_matrix, gate2_matrix)
			return qGate.UnitaryGate(combined_matrix, label=f'Unitary({name1},{name2})')
		else:
			return None  # Return None if one of the gates is an RZ gate

	def substitue_crosstalk(self, gate_dict):
		# qc = QuantumCircuit(2)
		qc = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		# Process gates by time slots
		for time in sorted(gate_dict.keys()):
			qubit0_gates = gate_dict[time][0]
			qubit1_gates = gate_dict[time][1]
			i = 0
			j = 0
			# print("time: ", time, qubit0_gates, qubit1_gates)
			cx_count = 0
			bar_count = 0
			temp_qubit = []

			while i < len(qubit0_gates) or j < len(qubit1_gates):
				gate0 = qubit0_gates[i] if i<len(qubit0_gates) else []
				# print("gate0: ", gate0, i)
				gate1 = qubit1_gates[j] if j<len(qubit1_gates) else []
				# print("gate1: ", gate1, j)
				if gate0:
					name0, params0, qubit0, index0 = gate0
				if gate1:
					name1, params1, qubit1, index1 = gate1
	
				if gate0 or gate1:
					temp_qubit =  qubit0 if len(qubit0) > 1 else qubit1
					if name1=='cx' and gate1:
						cx_count += 1
					if name0=='cx' and gate0:
						cx_count += 1
					if cx_count == 2:
						qc.append(qGate.CXGate(), temp_qubit)
						cx_count = 0
					if name1=='barrier' and gate1:
						bar_count += 1
					if name0=='barrier' and gate0:
						bar_count += 1
					if bar_count == 2:
						qc.append(qGate.Barrier(len(qubit1)), temp_qubit)
						bar_count = 0
					
				if gate0 and gate1:
					gates = []
					if name0 == 'sx':
						gates.append(qGate.SXGate())
					if name0 == 'x':
						gates.append(qGate.XGate())
					if name1 == 'sx':
						gates.append(qGate.SXGate())
					if name1 == 'x':
						gates.append(qGate.XGate())
					if len(gates)>1 and self.faultObject.isSameGateType(gates):
						unitary_gate = self.faultObject.createFaultyGate(self.faultObject.createOriginalGate())
						if unitary_gate:
							qc.append(unitary_gate, [qubit0[0], qubit1[0]])
					else:
						if name1 == 'sx':
							qc.append(qGate.SXGate(), qubit1)
						if name1 == 'rz':
							qc.append(qGate.RZGate(*params1), qubit1)	
						if name1 == 'x':
							qc.append(qGate.XGate(), qubit1)	
						if name0 == 'sx':
							qc.append(qGate.SXGate(), qubit0)
						if name0 == 'rz':
							qc.append(qGate.RZGate(*params0), qubit0)
						if name0 == 'x':
							qc.append(qGate.XGate(), qubit0)	
					i += 1
					j += 1
				elif gate0:
					# print('hello')
					if name0 == 'sx':
						qc.append(qGate.SXGate(), qubit0)
					if name0 == 'rz':
						qc.append(qGate.RZGate(*params0), qubit0)
					if name0 == 'x':
						qc.append(qGate.XGate(), qubit0)
					i += 1 
				elif gate1:
					# print('hello2')
					if name1 == 'sx':
						qc.append(qGate.SXGate(), qubit1)
					if name1 == 'rz':
						qc.append(qGate.RZGate(*params1), qubit1)	
					if name1 == 'x':
						qc.append(qGate.XGate(), qubit1)
					j += 1
		return qc

	def simulate(self):
		# simulateJob = execute(self.faultfreeQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		new_circuit = transpile(self.faultfreeQCKT, self.backend)
		durations = DynamicCircuitInstructionDurations.from_backend(self.backend)
		print(self.faultfreeQCKT, "=========================/n")
		print('nnnnn',new_circuit)
		# print(durations)
		# for instr, qargs, cargs in self.faultfreeQCKT.data:
		# 	print(instr)
		
		# dag = circuit_to_dag(new_circuit)
		# crosstalk_prop = {(0, 1) : {(2, 3) : 0.2, (2) : 0.15},
        #                     (4, 5) : {(2, 3) : 0.1},
        #                     (2, 3) : {(0, 1) : 0.05, (4, 5): 0.05}}
		# durations = DynamicCircuitInstructionDurations.from_backend(self.backend, unit='dt')
		# print(durations)
		# backend_prop = BackendProperties(self.backend)
		
			# self.faultyQCKT = qc

		# print(scheduled_circuit.data)
		# print(time)
		# self.add_delay(durations, concurrent_gates, scheduled_circuit)
		# print(concurrent_gates)

		# backend_prop = self.backend.properties()
		# pass_ = CrosstalkAdaptiveSchedule(backend_prop, crosstalk_prop, durations)
		# scheduled_dag = pass_.run(dag)
		# scheduled_circ = dag_to_circuit(scheduled_dag)

		# print(self.faultfreeQCKT)
		# print("=============================\n",self.faultyQCKT)
		simulateJob = self.backend.run(new_circuit, noise_model = self.noiseModel, seed_simulator = 1, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		self.faultfreeDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultfreeDistribution[int(k, 2)] = counts[k]
		self.faultfreeDistribution = np.array(self.faultfreeDistribution / np.sum(self.faultfreeDistribution))

		# simulateJob = execute(self.faultyQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		new_circuit = transpile(self.faultyQCKT, self.backend)
		# print('hihihi:', new_circuit)
		time_line, has_crosstalk  = get_simultaneuos_gates(new_circuit, self.backend)
		# print(time_line)
		if has_crosstalk:
			qc = self.substitue_crosstalk(time_line)
			if qc:
				qc.measure(self.quantumRegister, self.classicalRegister)
				self.faultyQCKT = qc
		print(self.faultyQCKT, "\n", self.faultfreeQCKT)
		new_circuit = transpile(self.faultyQCKT, self.backend)
		
		simulateJob = self.backend.run(new_circuit, noise_model = self.noiseModel, seed_simulator = 1, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		self.faultyDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultyDistribution[int(k, 2)] = counts[k]
		self.faultyDistribution = np.array(self.faultyDistribution / np.sum(self.faultyDistribution))

		self.repetition, self.boundary = self.calRepetition()

		self.simulatedOverkill = self.calOverkill()
		self.simulatedTestescape = self.calTestEscape()
		
		return

	def calRepetition(self):
		if self.faultfreeDistribution.shape != self.faultyDistribution.shape:
			raise ValueError('input shape not consistency')

		degreeOfFreedom = self.faultfreeDistribution.shape[0] - 1

		effectSize = qatgCalEffectSize(self.faultyDistribution, self.faultfreeDistribution)
		
		lowerBoundEffectSize = 0.8 if effectSize > 0.8 else effectSize

		chi2Value = chi2.ppf(self.targetAlpha, degreeOfFreedom)
		repetition = ceil(chi2Value / (lowerBoundEffectSize ** 2))
		nonCentrality = repetition * (effectSize ** 2)
		nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		while nonChi2Value < chi2Value:
			repetition += 1
			nonCentrality += effectSize ** 2
			nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		
		boundary = (nonChi2Value * 0.3 + chi2Value * 0.7)
		if repetition >= qatgINT_MAX or repetition <= 0:
			raise ValueError("Error occured calculating repetition")
		
		return repetition, boundary

	def calOverkill(self):
		overkill = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultfreeDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should pass, chiStatistic should > boundary
			if chiStatistic <= self.boundary:
				overkill += 1

		return overkill / self.testSampleTime

	def calTestEscape(self):
		testEscape = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultyDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should fail, chiStatistic should <= boundary
			if chiStatistic > self.boundary:
				testEscape += 1

		return testEscape / self.testSampleTime

	@property
	def circuit(self):
		return self.faultfreeQCKT
	
