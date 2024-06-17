from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
# from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passmanager import PassManager
# from qiskit.transpiler.passes import ALAPScheduleAnalysis, TimeUnitConversion
from qiskit_ibm_runtime.transpiler.passes.scheduling import DynamicCircuitInstructionDurations
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit.compiler import schedule
from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
# QiskitRuntimeService.save_account(channel="ibm_quantum", token='6be65f0ec78a1145f69c8dfeda2633401927ed71e9f1e918705862fcec493605c0743c8fa3dc97ad4dfb45472097a4acc3e6ad34b4abf26c8a97193065b62929')
# service = QiskitRuntimeService(instance="ibm-q/open/main")
# backend = service.backend('ibm_osaka')
# Create a quantum circuit
# qc = QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)
# qc.measure_all()


# qc = QuantumCircuit.from_qasm_file("benchmarks/qc3.qasm")

def get_simultaneuos_gates(qc, backend):
    # backend = FakeJakartaV2()

    # qc = transpile(qc, backend = backend)

    durations = DynamicCircuitInstructionDurations.from_backend(backend)
    # print(durations)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pm.scheduling = PassManager([ALAPScheduleAnalysis(durations)])
    for instr, qargs, _ in qc.data:
        qubit_indices = [q._index for q in qargs]
        duration = durations.get(instr.name, qubit_indices)
    # transpiled_circuit = transpile(qc, backend)
    scheduled_pulse = schedule(qc, backend=backend)
    scheduled_circuit = pm.run(qc)
    start_times = []
    data = []
    data_no_duration = []
    start_times_no_duration = []
    # print(scheduled_circuit, transpiled_circuit)
    for i in range(len(scheduled_circuit._op_start_times)):
        # data.append(scheduled_circuit.data[i])
        instr, qargs, cargs = scheduled_circuit.data[i]
        # idxs = []
        # for qubit in qargs:
        # if len(qargs) < 2:
            # idxs.append(qubit._index)
        if len(qargs) < 2:
            duration = durations.get(instr.name, qargs[0]._index)
            if not duration == 0 and not instr.name=='measure':
                data_no_duration.append((instr.name, instr.params, qargs[0]._index))
                start_times_no_duration.append(scheduled_circuit.op_start_times[i][1])
        qindex = []
        for qubits in qargs:
            qindex.append(qubits._index)
        duration = durations.get(instr.name, qindex)
        if not instr.name=='measure':
            data.append((instr.name, instr.params, qindex, i))
            start_times.append(scheduled_circuit.op_start_times[i][1])

    # print(start_times, data)
    start_time_groups = {}
    start_time_groups_no_duration = {}
    for i, start_time in enumerate(start_times):
        if start_time not in start_time_groups:
            start_time_groups[start_time] = []
        start_time_groups[start_time].append(data[i])

    for i, start_time in enumerate(start_times_no_duration):
        if start_time not in start_time_groups_no_duration:
            start_time_groups_no_duration[start_time] = []
        start_time_groups_no_duration[start_time].append(data_no_duration[i])

    concurrent_gates = []
    time_line = {}
    # print("start_time groups", start_time_groups)
    for start_time, ops in start_time_groups.items():
        qubit1 = []
        qubit0 = []
        for op in ops:
            if op[2]==[0]:
                qubit0.append(op)
            elif op[2]==[1]:
                qubit1.append(op)
            else:
                qubit1.append(op)
                qubit0.append(op)
        time_line[start_time] = [qubit0, qubit1]

    for start_time, ops in start_time_groups_no_duration.items():
        if len(ops) > 1:
            concurrent_gates.append((start_time, ops))

    # # print("Concurrent gates on qubits 0 and 1:", concurrent_gates)
    # has_crosstalk = False
    # for start_time, ops in concurrent_gates:
    #     # print(f"Time: {start_time}")

    #     for op in ops:
    #         qubits_used.append(op[2])
        
    return time_line, concurrent_gates


# service = QiskitRuntimeService(instance="ibm-q/open/main")
# backend = service.backend('ibm_brisbane')
# backend_properties = backend.properties()

# # # List gates and their properties
# # for gate in backend_properties.gates:
# #     print(f"Gate: {gate.gate}, Qubits: {gate.qubits}, Parameters: {gate.parameters}")

# # Check specific gate duration, e.g., 'rx' on qubit 0
# for gate in backend_properties.gates:
#     if gate.gate == 'rx' and 0 in gate.qubits:
#         print(f"Duration of 'rx' on qubit 0: {gate.parameters}")