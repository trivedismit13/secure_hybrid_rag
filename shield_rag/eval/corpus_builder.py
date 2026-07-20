"""
Synthetic equipment-manual corpus builder for SHIELD-RAG Phase 1 evaluation.

Generates a deterministic corpus of 300 passages about industrial centrifugal
pump maintenance, annotated with ontology-typed graph nodes and edges.
Includes 50+ true/false evaluation questions requiring multi-hop reasoning.

Usage:
    from shield_rag.eval.corpus_builder import build_corpus
    corpus = build_corpus()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from shield_rag.schema.ontology import (
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    num_passages: int = 300
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    answer: bool
    evidence_node_ids: list[str]
    question_type: str  # 'indicator_comparison', 'requirement_check', 'parameter_lookup'


@dataclass
class Corpus:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    questions: list[EvalQuestion]
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


# ---------------------------------------------------------------------------
# Seed data — realistic industrial centrifugal pump domain
# ---------------------------------------------------------------------------

# Top-level assemblies and their sub-components
_ASSEMBLIES: dict[str, list[str]] = {
    "Pump Assembly": [
        "Impeller Assembly",
        "Volute Casing",
        "Shaft Assembly",
        "Bearing Housing",
        "Mechanical Seal",
        "Coupling Guard",
        "Suction Nozzle",
        "Discharge Nozzle",
        "Wear Rings",
        "Baseplate",
    ],
    "Drive Train Assembly": [
        "Electric Motor",
        "Flexible Coupling",
        "Motor Mount",
        "Variable Frequency Drive",
        "Coupling Spacer",
        "Motor Bearing",
    ],
    "Lubrication System": [
        "Oil Reservoir",
        "Oil Pump",
        "Oil Filter",
        "Oil Cooler",
        "Sight Glass",
        "Pressure Relief Valve",
    ],
    "Seal Support System": [
        "Seal Flush Piping",
        "Seal Flush Filter",
        "Seal Flush Cooler",
        "Seal Quench Chamber",
        "Thermosyphon Loop",
    ],
    "Instrumentation Package": [
        "Vibration Sensor",
        "Temperature Probe",
        "Pressure Transmitter",
        "Flow Meter",
        "Level Switch",
        "Tachometer",
    ],
}

# Parameters for blocks — (param_name, value_text, unit)
_BLOCK_PARAMETERS: dict[str, list[tuple[str, str, str]]] = {
    "Impeller Assembly": [
        ("Diameter", "12", "inches"),
        ("Material", "316 Stainless Steel", ""),
        ("Number of Vanes", "6", ""),
        ("Max Tip Speed", "150", "ft/s"),
        ("Weight", "45", "lbs"),
    ],
    "Volute Casing": [
        ("Design Pressure", "500", "PSI"),
        ("Material", "Cast Duplex Stainless Steel", ""),
        ("Wall Thickness", "0.375", "inches"),
        ("Hydrostatic Test Pressure", "750", "PSI"),
    ],
    "Shaft Assembly": [
        ("Shaft Diameter", "3.5", "inches"),
        ("Material", "17-4 PH Stainless Steel", ""),
        ("Runout Tolerance", "0.002", "inches TIR"),
        ("Critical Speed", "4800", "RPM"),
    ],
    "Bearing Housing": [
        ("Bearing Type", "Double-row angular contact", ""),
        ("L10 Bearing Life", "40000", "hours"),
        ("Housing Material", "Cast Iron ASTM A48 Class 40", ""),
        ("Max Operating Temperature", "180", "°F"),
    ],
    "Mechanical Seal": [
        ("Seal Type", "Dual cartridge, API Plan 53B", ""),
        ("Face Material", "Silicon Carbide vs Carbon", ""),
        ("Max Pressure Rating", "400", "PSI"),
        ("Replacement Interval", "8000", "operating hours"),
    ],
    "Electric Motor": [
        ("Power Rating", "250", "HP"),
        ("Voltage", "460", "V"),
        ("Phase", "3", ""),
        ("Frame Size", "449T", "NEMA"),
        ("Full Load Amps", "302", "A"),
        ("Efficiency", "95.4", "%"),
    ],
    "Variable Frequency Drive": [
        ("Input Voltage Range", "380–500", "V"),
        ("Frequency Range", "0–120", "Hz"),
        ("Harmonic Distortion Limit", "5", "% THD"),
        ("Overload Capacity", "110", "% for 60s"),
    ],
    "Vibration Sensor": [
        ("Measurement Range", "0–25.4", "mm/s RMS"),
        ("Frequency Response", "2–10000", "Hz"),
        ("Sensitivity", "100", "mV/g"),
        ("Alert Threshold", "7.1", "mm/s RMS"),
        ("Alarm Threshold", "11.2", "mm/s RMS"),
    ],
    "Temperature Probe": [
        ("Type", "PT100 RTD", ""),
        ("Range", "-50 to 300", "°C"),
        ("Accuracy", "±0.15 + 0.002×t", "°C"),
    ],
    "Pressure Transmitter": [
        ("Range", "0–600", "PSI"),
        ("Accuracy", "±0.075", "% of span"),
        ("Output Signal", "4–20", "mA HART"),
    ],
    "Flow Meter": [
        ("Type", "Electromagnetic", ""),
        ("Max Flow Rate", "2500", "GPM"),
        ("Accuracy", "±0.5", "% of reading"),
        ("Pipe Size", "10", "inches"),
    ],
    "Oil Reservoir": [
        ("Capacity", "15", "gallons"),
        ("Material", "Carbon Steel with epoxy lining", ""),
    ],
    "Oil Filter": [
        ("Filtration Rating", "10", "micron"),
        ("Differential Pressure Alarm", "15", "PSI"),
    ],
    "Coupling Guard": [
        ("Material", "Perforated 14-gauge steel", ""),
        ("Guard Type", "OSHA-compliant full-enclosure", ""),
    ],
    "Wear Rings": [
        ("Clearance (new)", "0.012", "inches diametral"),
        ("Max Allowable Clearance", "0.024", "inches diametral"),
        ("Material", "Bronze CDA 932", ""),
    ],
    "Flexible Coupling": [
        ("Type", "Disc-pack", ""),
        ("Max Misalignment (angular)", "0.25", "degrees"),
        ("Max Misalignment (parallel)", "0.005", "inches"),
        ("Torque Rating", "5500", "ft-lbs"),
    ],
    "Seal Flush Cooler": [
        ("Cooling Medium", "Plant cooling water", ""),
        ("Design Duty", "5", "kW"),
        ("Max Inlet Temperature", "180", "°F"),
    ],
    "Baseplate": [
        ("Material", "Structural steel with epoxy grout", ""),
        ("Flatness Tolerance", "0.002", "inches/foot"),
    ],
}

# Operating parameters for the overall pump system
_SYSTEM_PARAMETERS: list[tuple[str, str, str]] = [
    ("Max Flow Rate", "2500", "GPM"),
    ("Rated Head", "350", "feet"),
    ("Operating Temperature Range", "-20 to 150", "°C"),
    ("Max Suction Pressure", "75", "PSI"),
    ("Max Discharge Pressure", "500", "PSI"),
    ("NPSH Required", "12", "feet"),
    ("Best Efficiency Point Flow", "2000", "GPM"),
    ("BEP Efficiency", "87", "%"),
    ("Minimum Continuous Flow", "500", "GPM"),
    ("Max Allowable Speed", "3600", "RPM"),
    ("Design Life", "20", "years"),
    ("Noise Level at 1m", "85", "dBA"),
]

# Safety requirements
_SAFETY_REQUIREMENTS: list[str] = [
    "Pump casing must withstand 500 PSI operating pressure without permanent deformation.",
    "All rotating components must be enclosed by OSHA-compliant guards during operation.",
    "Lock-Out/Tag-Out (LOTO) procedures must be completed before any maintenance activity.",
    "Emergency shutdown must bring pump to zero speed within 5 seconds of activation.",
    "Bearing housing temperature must not exceed 180°F during continuous operation.",
    "Vibration levels must remain below 7.1 mm/s RMS during normal operation.",
    "Coupling guard must be verified in place before pump start-up sequence.",
    "Hydrostatic pressure test must be performed at 1.5× design pressure (750 PSI).",
    "Personnel must wear PPE including safety glasses, steel-toe boots, and hearing protection when within 3 meters of operating pump.",
    "Suction and discharge valves must be verified open before starting pump to prevent dead-heading.",
    "Seal barrier fluid pressure must be maintained 25 PSI above process pressure at all times.",
    "Motor insulation resistance must measure above 100 MΩ at 500V DC before energizing.",
    "Alignment tolerance must be within 0.002 inches parallel and 0.001 inches/inch angular offset.",
    "All pressure-containing bolted joints must be torqued to specification per the flange management procedure.",
    "Minimum continuous flow must not drop below 500 GPM to prevent internal recirculation damage.",
    "Oil mist lubrication system must maintain 5–10 inches H2O header pressure.",
    "VFD output harmonic distortion must remain below 5% THD to prevent motor winding damage.",
    "Grounding continuity must be verified with resistance below 1 ohm before commissioning.",
    "Confined space entry permit required for volute casing internal inspection.",
    "Hot work permit required for any welding or grinding on pump skid.",
]

# Maintenance actions
_MAINTENANCE_ACTIONS: list[tuple[str, str]] = [
    ("Replace mechanical seal", "Replace mechanical seal every 8000 operating hours or upon detection of seal leakage exceeding 5 drops per minute."),
    ("Bearing inspection", "Inspect and re-grease bearings every 2000 operating hours. Replace bearings if vibration exceeds alarm threshold of 11.2 mm/s RMS."),
    ("Impeller clearance check", "Measure impeller wear ring clearance with feeler gauges during each planned outage. Replace wear rings when clearance exceeds 0.024 inches diametral."),
    ("Shaft alignment verification", "Perform laser alignment check after any maintenance that disturbs the coupling. Tolerance: 0.002 inches parallel, 0.001 inches/inch angular."),
    ("Oil change", "Drain and replace lubrication oil every 4000 operating hours or if oil analysis shows water content above 200 ppm or particle count exceeds ISO 18/16/13."),
    ("Vibration analysis", "Collect vibration data monthly using permanently installed accelerometers. Trend data to detect bearing defects, imbalance, and misalignment."),
    ("Motor insulation test", "Perform motor insulation resistance test (megger test) annually. Minimum acceptable reading is 100 MΩ at 500V DC."),
    ("Coupling inspection", "Inspect flexible coupling disc packs annually for cracks or fatigue. Replace entire disc pack set if any single disc shows damage."),
    ("Hydrostatic pressure test", "Perform hydrostatic test at 750 PSI (1.5× design pressure) after any casing repair or modification. Hold for 30 minutes with zero leakage."),
    ("Baseplate grouting inspection", "Inspect epoxy grout for cracks or voids annually. Repair any defects to maintain baseplate flatness within 0.002 inches/foot."),
    ("Suction strainer cleaning", "Clean suction strainer during each planned outage. Replace mesh if differential pressure exceeds 3 PSI at rated flow."),
    ("Discharge check valve test", "Test discharge check valve closure during each planned outage. Valve must seal with zero backflow within 2 seconds."),
    ("Motor bearing replacement", "Replace motor bearings at 40000 operating hours or when vibration data indicates inner race defect frequencies."),
    ("VFD filter replacement", "Replace VFD input line reactor filters every 12 months or when harmonic distortion exceeds 5% THD."),
    ("Seal flush filter change", "Replace seal flush filter element when differential pressure exceeds 15 PSI or every 3 months, whichever comes first."),
    ("Thermal imaging survey", "Conduct infrared thermography survey of motor, bearings, and electrical connections quarterly. Flag any hotspot exceeding 20°F above ambient baseline."),
    ("Foundation bolt torque check", "Verify foundation bolt torque annually using calibrated torque wrench. Re-torque if any bolt is found more than 10% below specification."),
    ("Impeller dynamic balancing", "Balance impeller to ISO 1940 G2.5 or better after any repair or modification. Maximum allowable residual unbalance: 4 g-mm/kg."),
    ("Pump performance test", "Conduct full pump performance test (head, flow, power, efficiency) after any major overhaul. Results must be within 3% of original test curve."),
    ("Corrosion thickness survey", "Perform ultrasonic thickness measurement on volute casing and piping annually. Minimum wall thickness: 0.250 inches for continued operation."),
]

# Extra passage templates for padding to 300
_EXTRA_PASSAGE_TEMPLATES: list[str] = [
    "When operating the {component} in ambient temperatures above 40°C, reduce maximum speed by 10% to prevent thermal stress on internal seals.",
    "The {component} requires a torque specification of {value} ft-lbs on the retaining bolts during reassembly.",
    "Spare parts inventory for the {component} should maintain a minimum of {value} units on-site for critical spares.",
    "Cleaning the {component} requires flushing with clean process-compatible solvent for a minimum of 15 minutes.",
    "Inspect {component} for signs of cavitation damage every 4000 operating hours — look for pitting on wetted surfaces.",
    "The {component} must be isolated using double-block-and-bleed valve arrangement before disassembly.",
    "Record all {component} maintenance activities in the CMMS with failure code classification per ISO 14224.",
    "The {component} operating manual revision {value} supersedes all prior revisions — destroy obsolete copies.",
    "Calibrate {component} sensors every 6 months using NIST-traceable reference standards.",
    "The {component} assembly drawing (DWG-{value}) must be referenced during disassembly to ensure correct reassembly sequence.",
    "After installing the {component}, perform a bump test to verify correct rotation direction before coupling to the pump.",
    "The recommended lubricant for the {component} is ISO VG 68 mineral oil meeting DIN 51517 Part 3 (CLP) specification.",
    "When shipping the {component} for off-site repair, secure it in a custom crate with desiccant bags to prevent moisture damage.",
    "The {component} is rated for Zone 2 (IEC 60079-10) hazardous area classification — verify ATEX/IECEx certification.",
    "Training on {component} maintenance requires completion of Competency Module CM-{value} before unsupervised work.",
    "The {component} installation must allow minimum clearance of 36 inches for personnel access and maintenance.",
    "Document the as-found condition of {component} with photographs before beginning any disassembly work.",
    "The {component} uses Viton O-rings rated for -20°C to 200°C — do not substitute with Buna-N in high-temperature service.",
    "The {component} must be preserved with VCI paper wrapping if idle for more than 30 days to prevent corrosion.",
    "Post-maintenance run-in for the {component} requires 4 hours at 50% load followed by 2 hours at full load.",
]


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _make_node_id(rng: random.Random) -> str:
    """Generate a deterministic UUID-style node id."""
    return f"{rng.randint(0x10000000, 0xFFFFFFFF):08x}-{rng.randint(0x1000, 0xFFFF):04x}-{rng.randint(0x1000, 0xFFFF):04x}-{rng.randint(0x1000, 0xFFFF):04x}-{rng.randint(0x100000000000, 0xFFFFFFFFFFFF):012x}"


def _build_blocks_and_edges(rng: random.Random) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Create Block nodes for assemblies + sub-components with PartOf edges."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    assembly_node_map: dict[str, str] = {}  # name -> node_id
    component_node_map: dict[str, str] = {}  # name -> node_id

    for assembly_name, sub_components in _ASSEMBLIES.items():
        asm_id = _make_node_id(rng)
        nodes.append(GraphNode(
            node_id=asm_id,
            node_type=NodeType.BLOCK,
            text=f"{assembly_name}: Top-level assembly in the centrifugal pump system. "
                 f"Contains {len(sub_components)} sub-components requiring integrated maintenance scheduling.",
            embedding=[],
        ))
        assembly_node_map[assembly_name] = asm_id

        for comp_name in sub_components:
            comp_id = _make_node_id(rng)
            nodes.append(GraphNode(
                node_id=comp_id,
                node_type=NodeType.BLOCK,
                text=f"{comp_name}: Sub-component of {assembly_name}. "
                     f"Refer to equipment manual section for detailed specifications and maintenance procedures.",
                embedding=[],
            ))
            component_node_map[comp_name] = comp_id
            # PartOf edge: sub-component -> assembly
            edges.append(GraphEdge(src_id=comp_id, dst_id=asm_id, relation=RelationType.PART_OF))

    return nodes, edges, assembly_node_map, component_node_map  # type: ignore[return-value]


def _build_parameters(
    rng: random.Random,
    component_node_map: dict[str, str],
    assembly_node_map: dict[str, str],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Create Parameter nodes and HasParameter edges."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    param_node_map: dict[str, str] = {}  # "component::param_name" -> node_id

    # Block-level parameters
    for comp_name, params in _BLOCK_PARAMETERS.items():
        comp_id = component_node_map.get(comp_name)
        if comp_id is None:
            continue
        for param_name, value, unit in params:
            pid = _make_node_id(rng)
            unit_str = f" {unit}" if unit else ""
            nodes.append(GraphNode(
                node_id=pid,
                node_type=NodeType.PARAMETER,
                text=f"{comp_name} — {param_name}: {value}{unit_str}",
                embedding=[],
            ))
            edges.append(GraphEdge(src_id=comp_id, dst_id=pid, relation=RelationType.HAS_PARAMETER))
            param_node_map[f"{comp_name}::{param_name}"] = pid

    # System-level parameters (attached to Pump Assembly)
    pump_id = assembly_node_map.get("Pump Assembly", "")
    for param_name, value, unit in _SYSTEM_PARAMETERS:
        pid = _make_node_id(rng)
        unit_str = f" {unit}" if unit else ""
        nodes.append(GraphNode(
            node_id=pid,
            node_type=NodeType.PARAMETER,
            text=f"Pump System — {param_name}: {value}{unit_str}",
            embedding=[],
        ))
        edges.append(GraphEdge(src_id=pump_id, dst_id=pid, relation=RelationType.HAS_PARAMETER))
        param_node_map[f"Pump System::{param_name}"] = pid

    return nodes, edges, param_node_map  # type: ignore[return-value]


def _build_requirements(
    rng: random.Random,
    assembly_node_map: dict[str, str],
    component_node_map: dict[str, str],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Create Requirement nodes and Satisfy edges (Block -> Requirement)."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    req_node_map: dict[int, str] = {}  # index -> node_id

    # Map requirements to satisfying blocks
    _req_satisfiers: list[list[str]] = [
        ["Volute Casing"],                                  # 0: 500 PSI
        ["Coupling Guard"],                                 # 1: rotating guards
        [],                                                 # 2: LOTO (procedural)
        ["Variable Frequency Drive", "Electric Motor"],     # 3: emergency shutdown
        ["Bearing Housing"],                                # 4: bearing temp
        ["Vibration Sensor"],                               # 5: vibration limit
        ["Coupling Guard"],                                 # 6: coupling guard verify
        ["Volute Casing"],                                  # 7: hydrostatic test
        [],                                                 # 8: PPE (procedural)
        ["Suction Nozzle", "Discharge Nozzle"],             # 9: valves open
        ["Mechanical Seal", "Seal Quench Chamber"],         # 10: barrier fluid
        ["Electric Motor"],                                 # 11: insulation resistance
        ["Flexible Coupling"],                              # 12: alignment tolerance
        [],                                                 # 13: bolted joints (procedural)
        ["Impeller Assembly"],                              # 14: minimum flow
        ["Oil Reservoir", "Oil Pump"],                      # 15: oil mist
        ["Variable Frequency Drive"],                       # 16: VFD harmonics
        [],                                                 # 17: grounding (procedural)
        ["Volute Casing"],                                  # 18: confined space
        [],                                                 # 19: hot work (procedural)
    ]

    for idx, req_text in enumerate(_SAFETY_REQUIREMENTS):
        rid = _make_node_id(rng)
        nodes.append(GraphNode(
            node_id=rid,
            node_type=NodeType.REQUIREMENT,
            text=f"REQ-{idx+1:03d}: {req_text}",
            embedding=[],
        ))
        req_node_map[idx] = rid

        satisfiers = _req_satisfiers[idx] if idx < len(_req_satisfiers) else []
        for comp_name in satisfiers:
            comp_id = component_node_map.get(comp_name) or assembly_node_map.get(comp_name)
            if comp_id:
                edges.append(GraphEdge(src_id=comp_id, dst_id=rid, relation=RelationType.SATISFY))

    return nodes, edges, req_node_map  # type: ignore[return-value]


def _build_actions(
    rng: random.Random,
    req_node_map: dict[int, str],
    component_node_map: dict[str, str],
    assembly_node_map: dict[str, str],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Create Action nodes, Trace edges (Requirement -> Action), and Allocate edges (Action -> Block)."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    action_node_map: dict[int, str] = {}

    # Map actions to requirements they trace from and blocks they allocate to
    _action_links: list[tuple[list[int], list[str]]] = [
        ([10], ["Mechanical Seal"]),                              # 0: replace seal
        ([4, 5], ["Bearing Housing", "Vibration Sensor"]),        # 1: bearing inspection
        ([14], ["Impeller Assembly", "Wear Rings"]),              # 2: impeller clearance
        ([12], ["Flexible Coupling", "Shaft Assembly"]),          # 3: shaft alignment
        ([15], ["Oil Reservoir", "Oil Filter"]),                  # 4: oil change
        ([5], ["Vibration Sensor"]),                              # 5: vibration analysis
        ([11], ["Electric Motor"]),                               # 6: motor insulation
        ([12], ["Flexible Coupling"]),                            # 7: coupling inspection
        ([0, 7], ["Volute Casing"]),                              # 8: hydrostatic test
        ([12], ["Baseplate"]),                                    # 9: baseplate grouting
        ([9], ["Suction Nozzle"]),                                # 10: suction strainer
        ([9], ["Discharge Nozzle"]),                              # 11: discharge check valve
        ([4], ["Motor Bearing", "Electric Motor"]),               # 12: motor bearing replace
        ([16], ["Variable Frequency Drive"]),                     # 13: VFD filter
        ([10], ["Seal Flush Filter"]),                            # 14: seal flush filter
        ([4], ["Electric Motor", "Bearing Housing"]),             # 15: thermal imaging
        ([17], ["Baseplate"]),                                    # 16: foundation bolts
        ([5, 14], ["Impeller Assembly"]),                         # 17: impeller balancing
        ([0], ["Impeller Assembly", "Volute Casing"]),            # 18: performance test
        ([0, 18], ["Volute Casing"]),                             # 19: corrosion survey
    ]

    for idx, (action_name, action_text) in enumerate(_MAINTENANCE_ACTIONS):
        aid = _make_node_id(rng)
        nodes.append(GraphNode(
            node_id=aid,
            node_type=NodeType.ACTION,
            text=f"ACT-{idx+1:03d} ({action_name}): {action_text}",
            embedding=[],
        ))
        action_node_map[idx] = aid

        # Trace edges: Requirement -> Action
        req_indices, block_names = _action_links[idx] if idx < len(_action_links) else ([], [])
        for ri in req_indices:
            if ri in req_node_map:
                edges.append(GraphEdge(src_id=req_node_map[ri], dst_id=aid, relation=RelationType.TRACE))

        # Allocate edges: Action -> Block
        for bname in block_names:
            bid = component_node_map.get(bname) or assembly_node_map.get(bname)
            if bid:
                edges.append(GraphEdge(src_id=aid, dst_id=bid, relation=RelationType.ALLOCATE))

    return nodes, edges, action_node_map  # type: ignore[return-value]


def _build_extra_passages(
    rng: random.Random,
    component_node_map: dict[str, str],
    current_count: int,
    target: int,
) -> list[GraphNode]:
    """Generate additional passages to reach the target node count."""
    nodes: list[GraphNode] = []
    components = list(component_node_map.keys())
    needed = target - current_count
    if needed <= 0:
        return nodes

    for i in range(needed):
        template = _EXTRA_PASSAGE_TEMPLATES[i % len(_EXTRA_PASSAGE_TEMPLATES)]
        comp = components[i % len(components)]
        value = str(rng.randint(2, 50))
        text = template.format(component=comp, value=value)
        nid = _make_node_id(rng)
        # Alternate node types for variety
        node_types = [NodeType.BLOCK, NodeType.ACTION, NodeType.REQUIREMENT, NodeType.PARAMETER]
        nt = node_types[i % 4]
        nodes.append(GraphNode(node_id=nid, node_type=nt, text=text, embedding=[]))

    return nodes


# ---------------------------------------------------------------------------
# Evaluation question generator
# ---------------------------------------------------------------------------

def _build_questions(
    rng: random.Random,
    req_node_map: dict[int, str],
    action_node_map: dict[int, str],
    component_node_map: dict[str, str],
    assembly_node_map: dict[str, str],
    param_node_map: dict[str, str],
) -> list[EvalQuestion]:
    """Generate 60 true/false evaluation questions requiring multi-hop graph reasoning."""
    questions: list[EvalQuestion] = []
    qid = 0

    def _q(question: str, answer: bool, evidence: list[str], qtype: str) -> None:
        nonlocal qid
        qid += 1
        questions.append(EvalQuestion(
            question_id=f"Q-{qid:03d}",
            question=question,
            answer=answer,
            evidence_node_ids=evidence,
            question_type=qtype,
        ))

    # Helpers
    def _nids(*keys: str) -> list[str]:
        result = []
        for k in keys:
            for m in [component_node_map, assembly_node_map, param_node_map]:
                if k in m:
                    result.append(m[k])
                    break
            # check req/action maps by index
        return result

    pump_asm = assembly_node_map.get("Pump Assembly", "")
    impeller = component_node_map.get("Impeller Assembly", "")
    volute = component_node_map.get("Volute Casing", "")
    mech_seal = component_node_map.get("Mechanical Seal", "")
    bearing = component_node_map.get("Bearing Housing", "")
    motor = component_node_map.get("Electric Motor", "")
    vfd = component_node_map.get("Variable Frequency Drive", "")
    coupling = component_node_map.get("Flexible Coupling", "")
    vib_sensor = component_node_map.get("Vibration Sensor", "")
    coupling_guard = component_node_map.get("Coupling Guard", "")
    wear_rings = component_node_map.get("Wear Rings", "")
    baseplate = component_node_map.get("Baseplate", "")
    oil_res = component_node_map.get("Oil Reservoir", "")
    oil_filter = component_node_map.get("Oil Filter", "")
    shaft = component_node_map.get("Shaft Assembly", "")
    seal_flush_filter = component_node_map.get("Seal Flush Filter", "")
    flow_meter = component_node_map.get("Flow Meter", "")
    temp_probe = component_node_map.get("Temperature Probe", "")
    press_tx = component_node_map.get("Pressure Transmitter", "")

    # --- indicator_comparison questions ---
    _q("Is the maximum operating pressure of the pump casing (500 PSI) greater than the mechanical seal pressure rating (400 PSI)?",
       True, [volute, mech_seal, param_node_map.get("Volute Casing::Design Pressure", ""),
              param_node_map.get("Mechanical Seal::Max Pressure Rating", "")], "indicator_comparison")

    _q("Is the impeller diameter (12 inches) larger than the shaft diameter (3.5 inches)?",
       True, [impeller, shaft, param_node_map.get("Impeller Assembly::Diameter", ""),
              param_node_map.get("Shaft Assembly::Shaft Diameter", "")], "indicator_comparison")

    _q("Is the hydrostatic test pressure (750 PSI) exactly 2× the design pressure (500 PSI)?",
       False, [volute, param_node_map.get("Volute Casing::Design Pressure", ""),
               param_node_map.get("Volute Casing::Hydrostatic Test Pressure", "")], "indicator_comparison")

    _q("Is the motor efficiency (95.4%) above the minimum threshold of 93%?",
       True, [motor, param_node_map.get("Electric Motor::Efficiency", "")], "indicator_comparison")

    _q("Is the vibration alert threshold (7.1 mm/s) higher than the alarm threshold (11.2 mm/s)?",
       False, [vib_sensor, param_node_map.get("Vibration Sensor::Alert Threshold", ""),
               param_node_map.get("Vibration Sensor::Alarm Threshold", "")], "indicator_comparison")

    _q("Is the L10 bearing life (40000 hours) longer than the mechanical seal replacement interval (8000 hours)?",
       True, [bearing, mech_seal, param_node_map.get("Bearing Housing::L10 Bearing Life", ""),
              param_node_map.get("Mechanical Seal::Replacement Interval", "")], "indicator_comparison")

    _q("Is the maximum flow rate (2500 GPM) more than 4× the minimum continuous flow (500 GPM)?",
       True, [pump_asm, param_node_map.get("Pump System::Max Flow Rate", ""),
              param_node_map.get("Pump System::Minimum Continuous Flow", "")], "indicator_comparison")

    _q("Is the shaft critical speed (4800 RPM) above the maximum allowable pump speed (3600 RPM)?",
       True, [shaft, param_node_map.get("Shaft Assembly::Critical Speed", ""),
              param_node_map.get("Pump System::Max Allowable Speed", "")], "indicator_comparison")

    _q("Is the oil filter differential pressure alarm (15 PSI) higher than the seal flush filter alarm (15 PSI)?",
       False, [oil_filter, seal_flush_filter, param_node_map.get("Oil Filter::Differential Pressure Alarm", ""),
               param_node_map.get("Seal Flush Filter::Differential Pressure Alarm", "" if "Seal Flush Filter::Differential Pressure Alarm" not in param_node_map else param_node_map["Seal Flush Filter::Differential Pressure Alarm"])], "indicator_comparison")

    _q("Is the motor power rating (250 HP) sufficient for a pump requiring approximately 200 HP at BEP?",
       True, [motor, param_node_map.get("Electric Motor::Power Rating", "")], "indicator_comparison")

    # --- requirement_check questions ---
    _q("Does REQ-001 (500 PSI casing pressure) require the Volute Casing to satisfy it?",
       True, [req_node_map.get(0, ""), volute], "requirement_check")

    _q("Is Lock-Out/Tag-Out (LOTO) required before maintenance activities per REQ-003?",
       True, [req_node_map.get(2, "")], "requirement_check")

    _q("Does the coupling guard need to be verified before pump start-up per REQ-007?",
       True, [req_node_map.get(6, ""), coupling_guard], "requirement_check")

    _q("Is the minimum motor insulation resistance requirement 50 MΩ per REQ-012?",
       False, [req_node_map.get(11, ""), motor], "requirement_check")

    _q("Does REQ-006 specify a vibration limit of 11.2 mm/s RMS for normal operation?",
       False, [req_node_map.get(5, ""), vib_sensor], "requirement_check")

    _q("Is a confined space entry permit required for volute casing internal inspection per REQ-019?",
       True, [req_node_map.get(18, ""), volute], "requirement_check")

    _q("Does REQ-011 require seal barrier fluid pressure to be maintained 25 PSI above process pressure?",
       True, [req_node_map.get(10, ""), mech_seal], "requirement_check")

    _q("Is PPE required when within 3 meters of an operating pump per REQ-009?",
       True, [req_node_map.get(8, "")], "requirement_check")

    _q("Does REQ-004 require the emergency shutdown to stop the pump within 10 seconds?",
       False, [req_node_map.get(3, ""), vfd, motor], "requirement_check")

    _q("Is the alignment tolerance 0.005 inches parallel per REQ-013?",
       False, [req_node_map.get(12, ""), coupling], "requirement_check")

    # --- parameter_lookup questions ---
    _q("Is the impeller made of 316 Stainless Steel?",
       True, [impeller, param_node_map.get("Impeller Assembly::Material", "")], "parameter_lookup")

    _q("Does the flow meter have an accuracy of ±1.0% of reading?",
       False, [flow_meter, param_node_map.get("Flow Meter::Accuracy", "")], "parameter_lookup")

    _q("Is the shaft runout tolerance 0.002 inches TIR?",
       True, [shaft, param_node_map.get("Shaft Assembly::Runout Tolerance", "")], "parameter_lookup")

    _q("Does the temperature probe use a thermocouple (Type K) as the sensing element?",
       False, [temp_probe, param_node_map.get("Temperature Probe::Type", "")], "parameter_lookup")

    _q("Is the pressure transmitter output signal 4-20 mA HART?",
       True, [press_tx, param_node_map.get("Pressure Transmitter::Output Signal", "")], "parameter_lookup")

    _q("Does the VFD have a harmonic distortion limit of 10% THD?",
       False, [vfd, param_node_map.get("Variable Frequency Drive::Harmonic Distortion Limit", "")], "parameter_lookup")

    _q("Is the wear ring initial clearance 0.012 inches diametral?",
       True, [wear_rings, param_node_map.get("Wear Rings::Clearance (new)", "")], "parameter_lookup")

    _q("Does the electric motor operate on single-phase power?",
       False, [motor, param_node_map.get("Electric Motor::Phase", "")], "parameter_lookup")

    _q("Is the BEP efficiency of the pump system 87%?",
       True, [pump_asm, param_node_map.get("Pump System::BEP Efficiency", "")], "parameter_lookup")

    _q("Is the baseplate flatness tolerance 0.005 inches/foot?",
       False, [baseplate, param_node_map.get("Baseplate::Flatness Tolerance", "")], "parameter_lookup")

    # --- Multi-hop reasoning questions ---
    _q("Does the Impeller Assembly (a sub-component of Pump Assembly) need to satisfy the 500 PSI casing requirement through its parent assembly?",
       True, [impeller, pump_asm, req_node_map.get(0, ""), volute], "requirement_check")

    _q("Can you trace from the Vibration Sensor back to the bearing temperature requirement (REQ-005) through the vibration monitoring action?",
       True, [vib_sensor, req_node_map.get(4, ""), action_node_map.get(1, "")], "requirement_check")

    _q("Is the mechanical seal replacement action (ACT-001) traceable to REQ-011 (seal barrier fluid)?",
       True, [action_node_map.get(0, ""), req_node_map.get(10, ""), mech_seal], "requirement_check")

    _q("Does the hydrostatic pressure test action (ACT-009) allocate to the Volute Casing?",
       True, [action_node_map.get(8, ""), volute], "requirement_check")

    _q("Is the shaft alignment action (ACT-004) allocated to the Impeller Assembly?",
       False, [action_node_map.get(3, ""), coupling, shaft], "requirement_check")

    _q("Does the Impeller Assembly have a parameter for maximum tip speed of 150 ft/s?",
       True, [impeller, param_node_map.get("Impeller Assembly::Max Tip Speed", "")], "parameter_lookup")

    _q("Is the oil change action (ACT-005) traceable to the oil mist lubrication requirement (REQ-016)?",
       True, [action_node_map.get(4, ""), req_node_map.get(15, "")], "requirement_check")

    _q("Does the VFD filter replacement action trace back to the VFD harmonics requirement (REQ-017)?",
       True, [action_node_map.get(13, ""), req_node_map.get(16, ""), vfd], "requirement_check")

    _q("Is the bearing inspection action (ACT-002) allocated to both the Bearing Housing and Vibration Sensor?",
       True, [action_node_map.get(1, ""), bearing, vib_sensor], "requirement_check")

    _q("Does the corrosion survey action (ACT-020) trace to REQ-001 (500 PSI pressure)?",
       True, [action_node_map.get(19, ""), req_node_map.get(0, ""), volute], "requirement_check")

    _q("Is the motor bearing replacement action (ACT-013) allocated to the Bearing Housing?",
       False, [action_node_map.get(12, ""), component_node_map.get("Motor Bearing", ""), motor], "requirement_check")

    _q("Does the impeller dynamic balancing action (ACT-018) trace to both REQ-006 (vibration) and REQ-015 (minimum flow)?",
       True, [action_node_map.get(17, ""), req_node_map.get(5, ""), req_node_map.get(14, "")], "requirement_check")

    _q("Is the Seal Flush Filter part of the Seal Support System assembly?",
       True, [seal_flush_filter, assembly_node_map.get("Seal Support System", "")], "requirement_check")

    _q("Is the Vibration Sensor part of the Drive Train Assembly?",
       False, [vib_sensor, assembly_node_map.get("Instrumentation Package", "")], "requirement_check")

    _q("Does the Pump Assembly have a parameter for NPSH Required of 12 feet?",
       True, [pump_asm, param_node_map.get("Pump System::NPSH Required", "")], "parameter_lookup")

    _q("Is the flexible coupling torque rating (5500 ft-lbs) a parameter of the Drive Train Assembly?",
       False, [coupling, param_node_map.get("Flexible Coupling::Torque Rating", "")], "parameter_lookup")

    _q("Can the requirement for emergency shutdown within 5 seconds (REQ-004) be satisfied by the VFD and Electric Motor?",
       True, [req_node_map.get(3, ""), vfd, motor], "requirement_check")

    _q("Does the pump performance test action (ACT-019) allocate to both the Impeller Assembly and Volute Casing?",
       True, [action_node_map.get(18, ""), impeller, volute], "requirement_check")

    _q("Is the suction strainer cleaning action (ACT-011) traceable to the requirement about verifying suction valves (REQ-010)?",
       True, [action_node_map.get(10, ""), req_node_map.get(9, "")], "requirement_check")

    _q("Does the thermal imaging survey action (ACT-016) allocate to the Volute Casing?",
       False, [action_node_map.get(15, ""), motor, bearing], "requirement_check")

    _q("Is the Sight Glass a sub-component of the Lubrication System?",
       True, [component_node_map.get("Sight Glass", ""), assembly_node_map.get("Lubrication System", "")], "requirement_check")

    _q("Does the discharge check valve test action trace to the same requirement as the suction strainer cleaning?",
       True, [action_node_map.get(11, ""), action_node_map.get(10, ""), req_node_map.get(9, "")], "requirement_check")

    _q("Is the rated head of the pump system 450 feet?",
       False, [pump_asm, param_node_map.get("Pump System::Rated Head", "")], "parameter_lookup")

    _q("Does the noise level parameter (85 dBA at 1m) exceed the OSHA 8-hour permissible exposure limit of 90 dBA?",
       False, [pump_asm, param_node_map.get("Pump System::Noise Level at 1m", "")], "indicator_comparison")

    _q("Is the maximum allowable wear ring clearance (0.024 inches) exactly double the new clearance (0.012 inches)?",
       True, [wear_rings, param_node_map.get("Wear Rings::Clearance (new)", ""),
              param_node_map.get("Wear Rings::Max Allowable Clearance", "")], "indicator_comparison")

    _q("Does the foundation bolt torque check action (ACT-017) trace to the grounding requirement (REQ-018)?",
       True, [action_node_map.get(16, ""), req_node_map.get(17, "")], "requirement_check")

    _q("Is the Coupling Spacer a sub-component of the Pump Assembly?",
       False, [component_node_map.get("Coupling Spacer", ""), assembly_node_map.get("Drive Train Assembly", "")], "requirement_check")

    _q("Does the design life parameter (20 years) belong to the Pump System?",
       True, [pump_asm, param_node_map.get("Pump System::Design Life", "")], "parameter_lookup")

    _q("Is the bearing housing material Cast Iron ASTM A48 Class 30?",
       False, [bearing, param_node_map.get("Bearing Housing::Housing Material", "")], "parameter_lookup")

    _q("Is the motor full load current 302 A?",
       True, [motor, param_node_map.get("Electric Motor::Full Load Amps", "")], "parameter_lookup")

    # Clean up empty evidence ids
    for q in questions:
        q.evidence_node_ids = [eid for eid in q.evidence_node_ids if eid]

    return questions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_corpus(config: CorpusConfig | None = None) -> Corpus:
    """Build the complete synthetic corpus with deterministic seeding."""
    if config is None:
        config = CorpusConfig()

    rng = random.Random(config.seed)

    # Phase 1: Build structured graph
    block_nodes, part_of_edges, assembly_map, component_map = _build_blocks_and_edges(rng)
    param_nodes, param_edges, param_map = _build_parameters(rng, component_map, assembly_map)
    req_nodes, req_edges, req_map = _build_requirements(rng, assembly_map, component_map)
    action_nodes, action_edges, action_map = _build_actions(rng, req_map, component_map, assembly_map)

    # Combine all structured nodes
    all_nodes = block_nodes + param_nodes + req_nodes + action_nodes
    all_edges = part_of_edges + param_edges + req_edges + action_edges

    # Phase 2: Pad with extra passages to reach target count
    extra_nodes = _build_extra_passages(rng, component_map, len(all_nodes), config.num_passages)
    all_nodes.extend(extra_nodes)

    # Phase 3: Build evaluation questions
    questions = _build_questions(rng, req_map, action_map, component_map, assembly_map, param_map)

    # Phase 4: Split into train/val/test
    node_ids = [n.node_id for n in all_nodes]
    shuffled_ids = list(node_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    return Corpus(
        nodes=all_nodes,
        edges=all_edges,
        questions=questions,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
