"""Manufacturing domain knowledge for MVTec AD's defect taxonomy.

MVTec AD labels *what* a defect looks like ("crack", "bent_lead") but says
nothing about *why* it happened - it is a vision benchmark, not a process
dataset. Root-cause reasoning needs that missing half, so this module supplies
it: for each of the 73 (category, defect_type) pairs, the manufacturing process
step it originates in, the causes an experienced process engineer would consider,
and the corrective actions they would take.

**Provenance, stated plainly.** This is synthesised engineering knowledge, not
records from a real production line. The causes are drawn from standard failure
modes for the relevant processes (injection moulding, tablet compression, cold
forming, wire extrusion, ceramic firing) and are individually plausible, but no
claim is made that they reflect the true history of the MVTec samples - such
history does not exist publicly. Retrieval metrics computed over the corpus
therefore measure whether the *mechanism* works, not field accuracy. This is
restated in EVALUATION.md and the report's Limitations section.

Ordering matters: within each entry, causes are listed most-likely first, and the
corpus generator samples them with a matching bias, so the resulting case
distribution mirrors real Pareto-shaped failure data rather than a uniform mix.
"""

from __future__ import annotations

from dataclasses import dataclass

from mavia.schemas import RiskLevel


@dataclass(frozen=True)
class DefectKnowledge:
    """What a process engineer knows about one defect mode."""

    process_step: str
    root_causes: tuple[str, ...]
    actions: tuple[str, ...]
    severity: RiskLevel
    description: str


# Shorthand for readability below.
LOW, MED, HIGH, CRIT = RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL


KNOWLEDGE_BASE: dict[tuple[str, str], DefectKnowledge] = {
    # ---------------------------------------------------------------- bottle
    ("bottle", "broken_large"): DefectKnowledge(
        "glass forming / annealing",
        (
            "Thermal shock from an excessive cooling gradient in the annealing lehr",
            "Mould wear producing thin sidewall sections that fail under internal pressure",
            "Mechanical impact during transfer from the forming machine to the conveyor",
        ),
        (
            "Re-profile the annealing lehr temperature curve and verify with a thermal probe",
            "Inspect and replace the affected mould cavity; log cavity cycle count",
            "Reduce transfer conveyor speed and add guide-rail padding",
        ),
        CRIT,
        "Large fracture through the bottle wall, structurally unsound",
    ),
    ("bottle", "broken_small"): DefectKnowledge(
        "glass forming / handling",
        (
            "Chipping at the finish from bottle-to-bottle contact on the accumulation table",
            "Localised mould defect creating a stress concentration point",
            "Over-pressure during the blow cycle thinning the shoulder",
        ),
        (
            "Increase line pressure control to reduce bottle contact on the accumulator",
            "Polish or replace the mould cavity showing the surface imperfection",
            "Recalibrate blow pressure to the validated setpoint",
        ),
        HIGH,
        "Small chip or fracture, typically at the rim or shoulder",
    ),
    ("bottle", "contamination"): DefectKnowledge(
        "resin handling / moulding",
        (
            "Foreign particulate in the resin feed from a degraded hopper filter",
            "Carbon deposits shedding from an over-temperature barrel",
            "Airborne contamination from inadequate cleanroom differential pressure",
        ),
        (
            "Replace the hopper inlet filter and purge the feed line",
            "Purge the barrel and verify melt-temperature profile against spec",
            "Verify cleanroom pressure differential and HEPA filter integrity",
        ),
        HIGH,
        "Foreign material embedded in or adhering to the bottle wall",
    ),
    # ----------------------------------------------------------------- cable
    ("cable", "bent_wire"): DefectKnowledge(
        "wire feeding / assembly",
        (
            "Excessive back-tension in the wire feeder deforming the conductor",
            "Operator handling damage during manual routing",
            "Guide-roller misalignment imposing a lateral load on the wire",
        ),
        (
            "Reduce feeder back-tension to the validated range and re-verify",
            "Retrain the assembly operator on the routing work instruction",
            "Realign the guide rollers and check runout",
        ),
        MED,
        "Conductor deformed from its intended straight path",
    ),
    ("cable", "cable_swap"): DefectKnowledge(
        "assembly / wiring",
        (
            "Colour-code confusion during manual termination",
            "Incorrect fixture setup after a product changeover",
            "Work instruction revision not deployed to the station",
        ),
        (
            "Add poka-yoke keyed connectors to make the swap physically impossible",
            "Add a changeover verification step with first-article inspection",
            "Audit the document control process for work-instruction deployment",
        ),
        CRIT,
        "Conductors terminated in the wrong positions - a functional safety risk",
    ),
    ("cable", "combined"): DefectKnowledge(
        "assembly",
        (
            "Systematic process drift affecting several parameters at once",
            "A single upstream fault cascading into multiple downstream symptoms",
            "Operation continued after a machine fault rather than stopping the line",
        ),
        (
            "Stop the line and perform a full process audit before resuming",
            "Trace the upstream fault rather than treating each symptom separately",
            "Review the fault-response procedure and escalation authority",
        ),
        CRIT,
        "Multiple co-occurring defects, indicating a process out of control",
    ),
    ("cable", "cut_inner_insulation"): DefectKnowledge(
        "stripping",
        (
            "Worn stripping blade cutting beyond the outer jacket",
            "Stripping depth set beyond the validated value",
            "Wire diameter variation outside the tolerance the setup assumes",
        ),
        (
            "Replace the stripping blade and reset the cycle counter",
            "Recalibrate stripping depth and add first-article verification",
            "Tighten incoming wire diameter inspection at goods-in",
        ),
        HIGH,
        "Inner insulation breached, exposing conductor - dielectric risk",
    ),
    ("cable", "cut_outer_insulation"): DefectKnowledge(
        "stripping / handling",
        (
            "Blade wear on the outer jacket stripper",
            "Sharp edge on a fixture or guide abrading the jacket in transit",
            "Excessive pull force drawing the cable across a hard edge",
        ),
        (
            "Replace the stripping blade per the preventive maintenance schedule",
            "Deburr and radius the fixture edges on the transfer path",
            "Reduce pull force and add a low-friction guide",
        ),
        MED,
        "Outer jacket cut or abraded without exposing the conductor",
    ),
    ("cable", "missing_cable"): DefectKnowledge(
        "assembly",
        (
            "Component feeder jam leaving the position unpopulated",
            "Operator omission at a manual insertion station",
            "Kitting error upstream, delivering an incomplete component set",
        ),
        (
            "Clear the feeder and add a jam-detection sensor with line stop",
            "Add a presence-check vision station after assembly",
            "Audit the kitting process and add a count verification",
        ),
        CRIT,
        "An entire cable absent from the assembly",
    ),
    ("cable", "missing_wire"): DefectKnowledge(
        "assembly / termination",
        (
            "Individual conductor not seated during termination",
            "Wire pulled free by insufficient crimp retention force",
            "Miscount at the manual preparation step",
        ),
        (
            "Add a pull-test sampling plan on crimp retention",
            "Recalibrate the crimp tool and verify crimp height",
            "Add a conductor count check to the station's work instruction",
        ),
        CRIT,
        "A conductor missing from an otherwise complete assembly",
    ),
    ("cable", "poke_insulation"): DefectKnowledge(
        "handling / test",
        (
            "Test probe penetrating the insulation during continuity check",
            "Sharp tooling contacting the cable during handling",
            "Over-travel of an automated gripper",
        ),
        (
            "Replace test probes with lower-force spring-loaded contacts",
            "Deburr tooling and add protective sheathing at contact points",
            "Recalibrate gripper travel limits",
        ),
        HIGH,
        "Puncture through the insulation, typically a small round breach",
    ),
    # --------------------------------------------------------------- capsule
    ("capsule", "crack"): DefectKnowledge(
        "capsule filling / sealing",
        (
            "Over-compression at the capsule closing station",
            "Gelatin shell embrittlement from low storage humidity",
            "Mechanical shock in the transfer chute",
        ),
        (
            "Reduce closing-station compression force and re-validate",
            "Restore shell storage humidity to the 35-55% RH specification",
            "Add cushioning to the transfer chute and reduce the drop height",
        ),
        HIGH,
        "Fracture in the capsule shell, risking dose loss",
    ),
    ("capsule", "faulty_imprint"): DefectKnowledge(
        "printing",
        (
            "Ink viscosity drift from solvent evaporation in an open reservoir",
            "Worn or contaminated print roller",
            "Capsule misalignment in the print carrier",
        ),
        (
            "Implement closed-reservoir ink handling with viscosity monitoring",
            "Clean or replace the print roller and log the cycle count",
            "Realign the print carrier and verify with a first-article check",
        ),
        MED,
        "Imprint smeared, incomplete, or illegible - a traceability failure",
    ),
    ("capsule", "poke"): DefectKnowledge(
        "handling",
        (
            "Vacuum pick nozzle pressure indenting the shell",
            "Sharp point contact in a transfer fixture",
            "Excessive gripper closing force",
        ),
        (
            "Reduce vacuum pick pressure to the validated setpoint",
            "Radius the contact points on the transfer fixture",
            "Recalibrate gripper force and add a force-monitoring check",
        ),
        MED,
        "Localised indentation or puncture in the shell",
    ),
    ("capsule", "scratch"): DefectKnowledge(
        "handling / conveying",
        (
            "Abrasive contact with a worn conveyor guide rail",
            "Capsule-to-capsule abrasion in an over-full accumulator",
            "Debris trapped in the transfer track",
        ),
        (
            "Replace the worn guide rail with a low-friction polymer insert",
            "Reduce accumulator fill level and improve flow control",
            "Add a scheduled track cleaning step to the changeover procedure",
        ),
        LOW,
        "Surface abrasion on the shell without breaching it",
    ),
    ("capsule", "squeeze"): DefectKnowledge(
        "filling / closing",
        (
            "Excessive gripper or closing-station clamping force",
            "Capsule body and cap size mismatch after a component change",
            "Shell softening from elevated process temperature",
        ),
        (
            "Reduce clamping force and verify against the capsule size spec",
            "Verify incoming shell dimensions at goods-in after any supplier change",
            "Restore the process temperature setpoint and check the chiller",
        ),
        MED,
        "Capsule deformed out of round by compression",
    ),
    # ---------------------------------------------------------------- carpet
    ("carpet", "color"): DefectKnowledge(
        "dyeing",
        (
            "Dye lot variation between batches without adequate blending",
            "Uneven dye bath temperature producing inconsistent uptake",
            "pH drift in the dye bath altering fixation",
        ),
        (
            "Enforce dye lot blending and add spectrophotometric batch release",
            "Service the dye bath circulation and verify temperature uniformity",
            "Add continuous pH monitoring with automated correction",
        ),
        MED,
        "Colour deviation from the reference standard",
    ),
    ("carpet", "cut"): DefectKnowledge(
        "shearing / finishing",
        (
            "Shearing blade set too low, cutting into the backing",
            "Blade nick producing a repeating linear defect",
            "Fabric tension loss allowing the pile to lift into the blade",
        ),
        (
            "Reset shearing height and verify with a pile-height gauge",
            "Replace the sheared blade and inspect for nicks each shift",
            "Restore web tension control and check the tensioner load cell",
        ),
        HIGH,
        "Linear cut through pile or backing",
    ),
    ("carpet", "hole"): DefectKnowledge(
        "tufting",
        (
            "Broken tufting needle tearing the primary backing",
            "Backing material weakness from an incoming quality defect",
            "Excessive tufting penetration depth",
        ),
        (
            "Replace the broken needle and inspect the full needle bar",
            "Tighten incoming backing inspection and notify the supplier",
            "Reset penetration depth to the validated value",
        ),
        HIGH,
        "Void through the carpet structure",
    ),
    ("carpet", "metal_contamination"): DefectKnowledge(
        "tufting / conveying",
        (
            "Needle fragment retained in the pile after a needle break",
            "Wear debris from a degraded machine bearing",
            "Foreign metal introduced with the raw fibre",
        ),
        (
            "Add inline metal detection after tufting with automatic reject",
            "Replace the worn bearing and add vibration monitoring",
            "Add a metal-detection check at fibre goods-in",
        ),
        CRIT,
        "Metallic foreign body in the carpet - an end-user injury hazard",
    ),
    ("carpet", "thread"): DefectKnowledge(
        "tufting",
        (
            "Yarn end not trimmed after a creel splice",
            "Thread break leaving a loose tail in the pile",
            "Inadequate shearing at the finishing stage",
        ),
        (
            "Add a post-splice trim verification to the creel changeover",
            "Improve yarn tension control to reduce break frequency",
            "Verify shearing coverage across the full web width",
        ),
        LOW,
        "Loose or protruding thread on the carpet surface",
    ),
    # ------------------------------------------------------------------ grid
    ("grid", "bent"): DefectKnowledge(
        "forming / handling",
        (
            "Impact or point load during stacking and transport",
            "Forming press misalignment applying uneven load",
            "Insufficient support fixturing during downstream handling",
        ),
        (
            "Introduce interleaved stacking separators and cap stack height",
            "Realign the forming press and verify die parallelism",
            "Redesign the handling fixture to support the full grid area",
        ),
        MED,
        "Grid wires deflected out of the intended plane",
    ),
    ("grid", "broken"): DefectKnowledge(
        "welding / forming",
        (
            "Weak spot weld failing under handling load",
            "Wire embrittlement from excessive weld heat input",
            "Material fatigue from repeated flexing in the line",
        ),
        (
            "Recalibrate weld current and add destructive pull testing",
            "Reduce weld energy and verify with metallurgical section review",
            "Reduce line-induced flexing and add support rollers",
        ),
        HIGH,
        "Fractured or severed grid wire",
    ),
    ("grid", "glue"): DefectKnowledge(
        "bonding",
        (
            "Adhesive over-application from a worn dispensing nozzle",
            "Adhesive squeeze-out from excessive clamping pressure",
            "Viscosity drift from adhesive temperature variation",
        ),
        (
            "Replace the dispensing nozzle and recalibrate the shot volume",
            "Reduce clamping pressure to the validated setpoint",
            "Add adhesive temperature conditioning and viscosity checks",
        ),
        LOW,
        "Excess adhesive residue on the grid surface",
    ),
    ("grid", "metal_contamination"): DefectKnowledge(
        "cutting / welding",
        (
            "Weld spatter adhering to the grid surface",
            "Cutting swarf not removed before the next operation",
            "Tool wear debris from the forming die",
        ),
        (
            "Add anti-spatter treatment and post-weld cleaning",
            "Introduce a deburr and wash step after cutting",
            "Replace the worn die and add particle monitoring",
        ),
        HIGH,
        "Metallic debris attached to the grid",
    ),
    ("grid", "thread"): DefectKnowledge(
        "handling / environment",
        (
            "Textile fibre contamination from operator clothing or wipes",
            "Airborne fibre settling on the product before packing",
            "Packaging material shedding fibres",
        ),
        (
            "Switch to lint-free wipes and specify appropriate line garments",
            "Improve air filtration and add a pre-pack air-knife blow-off",
            "Qualify a low-shed packaging material",
        ),
        LOW,
        "Textile fibre adhering to the grid",
    ),
    # -------------------------------------------------------------- hazelnut
    ("hazelnut", "crack"): DefectKnowledge(
        "harvest / drying / sorting",
        (
            "Over-aggressive drying producing shell thermal stress",
            "Mechanical impact in the sorting conveyor drop",
            "Excessive roller pressure at the grading station",
        ),
        (
            "Reduce the drying ramp rate and hold within the validated profile",
            "Lower conveyor drop heights and add cushioned transitions",
            "Reduce grading roller pressure and verify with a sample check",
        ),
        MED,
        "Shell fracture, creating a spoilage and contamination path",
    ),
    ("hazelnut", "cut"): DefectKnowledge(
        "processing / handling",
        (
            "Sharp edge on processing equipment scoring the shell",
            "Damage from a cutting or sizing blade during grading",
            "Abrasion against a worn chute liner",
        ),
        (
            "Deburr and radius the equipment contact surfaces",
            "Reposition the sizing blade clearance and verify",
            "Replace the chute liner and add it to the PM schedule",
        ),
        MED,
        "Linear incision in the shell surface",
    ),
    ("hazelnut", "hole"): DefectKnowledge(
        "raw material / pest control",
        (
            "Insect boring damage originating in the orchard or store",
            "Mechanical puncture from foreign material in the line",
            "Localised shell weakness failing under handling load",
        ),
        (
            "Review store fumigation and pest-monitoring records",
            "Add foreign-object detection ahead of grading",
            "Tighten incoming raw-material acceptance criteria",
        ),
        HIGH,
        "Perforation through the shell - a food safety concern",
    ),
    ("hazelnut", "print"): DefectKnowledge(
        "marking / handling",
        (
            "Ink transfer from marked product in a shared chute",
            "Residue from a lubricant or marking fluid on the equipment",
            "Contact with printed packaging material before drying",
        ),
        (
            "Segregate marked and unmarked product streams",
            "Switch to food-grade lubricant and add a cleaning verification",
            "Extend the ink cure time before packaging contact",
        ),
        LOW,
        "Unintended ink or residue marking on the shell",
    ),
    # --------------------------------------------------------------- leather
    ("leather", "color"): DefectKnowledge(
        "tanning / dyeing",
        (
            "Uneven dye penetration from hide thickness variation",
            "Drum rotation speed variation producing inconsistent agitation",
            "Batch-to-batch dye concentration drift",
        ),
        (
            "Sort hides by thickness before the dye batch",
            "Service the drum drive and verify rotation speed consistency",
            "Add gravimetric dye dosing with batch release testing",
        ),
        MED,
        "Colour non-uniformity across the hide",
    ),
    ("leather", "cut"): DefectKnowledge(
        "cutting / handling",
        (
            "Blade over-travel at the cutting station",
            "Pre-existing hide damage from flaying not caught at inspection",
            "Sharp fixture edge scoring the surface during transfer",
        ),
        (
            "Reset cutting depth and add a first-article verification",
            "Strengthen hide grading at goods-in and feed back to the supplier",
            "Radius the fixture edges on the transfer path",
        ),
        HIGH,
        "Incision through the leather surface",
    ),
    ("leather", "fold"): DefectKnowledge(
        "drying / storage",
        (
            "Improper hanging during drying, setting a permanent crease",
            "Compressive storage stacking beyond the recommended height",
            "Insufficient conditioning before the flattening operation",
        ),
        (
            "Revise the drying rack layout to eliminate contact folds",
            "Cap stack height and introduce interleaving separators",
            "Extend the conditioning dwell before flattening",
        ),
        LOW,
        "Permanent crease or fold line in the material",
    ),
    ("leather", "glue"): DefectKnowledge(
        "bonding / finishing",
        (
            "Adhesive squeeze-out at a lamination step",
            "Over-application from a miscalibrated dispenser",
            "Contact with adhesive residue on the work surface",
        ),
        (
            "Reduce lamination pressure and recalibrate the adhesive bead",
            "Recalibrate the dispenser shot volume",
            "Add a work-surface cleaning step between batches",
        ),
        LOW,
        "Adhesive residue visible on the finished surface",
    ),
    ("leather", "poke"): DefectKnowledge(
        "handling / fixturing",
        (
            "Pin or clamp point-loading the surface during fixturing",
            "Puncture from a stapling or tacking operation",
            "Foreign object pressed into the surface under a roller",
        ),
        (
            "Replace pin clamps with distributed-pressure vacuum fixturing",
            "Reposition tacking points into the trim allowance",
            "Add a roller-path inspection to the shift start-up checklist",
        ),
        MED,
        "Puncture or deep indentation in the surface",
    ),
    # ------------------------------------------------------------- metal_nut
    ("metal_nut", "bent"): DefectKnowledge(
        "stamping / forming",
        (
            "Press force exceeding the validated setpoint",
            "Die misalignment producing asymmetric loading",
            "Ejection damage as the part clears the die",
        ),
        (
            "Reset press tonnage and verify with a load-cell reading",
            "Realign the die set and check shut height and parallelism",
            "Adjust ejector timing and stroke",
        ),
        HIGH,
        "Part deformed out of its intended flat geometry",
    ),
    ("metal_nut", "color"): DefectKnowledge(
        "heat treatment / surface finishing",
        (
            "Over-temperature in heat treatment causing surface oxidation",
            "Anodising or plating bath concentration drift",
            "Inadequate rinse leaving a chemical residue stain",
        ),
        (
            "Recalibrate the furnace controller and verify with a thermocouple survey",
            "Add bath titration on a fixed schedule with automated dosing",
            "Increase rinse dwell time and monitor rinse conductivity",
        ),
        MED,
        "Surface discoloration from thermal or chemical process deviation",
    ),
    ("metal_nut", "flip"): DefectKnowledge(
        "feeding / orientation",
        (
            "Bowl feeder orientation track allowing inverted parts through",
            "Feeder vibration amplitude drift changing part behaviour",
            "Orientation sensor fault failing to reject inverted parts",
        ),
        (
            "Re-tune the bowl feeder orientation tooling and verify at rate",
            "Reset the feeder amplitude and add drift monitoring",
            "Replace the orientation sensor and verify the reject mechanism",
        ),
        MED,
        "Part presented inverted - an assembly orientation failure",
    ),
    ("metal_nut", "scratch"): DefectKnowledge(
        "machining / handling",
        (
            "Tool wear producing surface scoring during machining",
            "Part-to-part abrasion in bulk transport containers",
            "Chip or swarf trapped between the part and fixture",
        ),
        (
            "Replace the cutting tool per its wear schedule and log tool life",
            "Introduce dividers or dunnage in transport containers",
            "Add a chip-clearing air blast before fixture loading",
        ),
        LOW,
        "Surface scoring without dimensional impact",
    ),
    # ------------------------------------------------------------------ pill
    ("pill", "color"): DefectKnowledge(
        "blending / coating",
        (
            "Inadequate blend uniformity leaving colourant poorly dispersed",
            "Coating pan spray-rate variation producing uneven film build",
            "Colourant lot variation between batches",
        ),
        (
            "Extend blend time and validate with stratified content-uniformity sampling",
            "Recalibrate the coating spray rate and pan rotation speed",
            "Add incoming colourant lot verification against a reference",
        ),
        MED,
        "Colour deviation from the approved product appearance",
    ),
    ("pill", "combined"): DefectKnowledge(
        "compression / coating",
        (
            "Multiple parameters drifting together, indicating loss of process control",
            "An upstream granulation problem cascading through the line",
            "Continued operation despite an out-of-specification trend",
        ),
        (
            "Stop the line and perform a full batch investigation before release",
            "Investigate granulation properties rather than the symptoms downstream",
            "Review the trend-response procedure and operator escalation authority",
        ),
        CRIT,
        "Several defect modes present together - the batch is suspect",
    ),
    ("pill", "contamination"): DefectKnowledge(
        "compression / environment",
        (
            "Cross-contamination from inadequate cleaning between products",
            "Lubricant or oil ingress from the tablet press",
            "Airborne particulate from insufficient room air classification",
        ),
        (
            "Revalidate the cleaning procedure with swab recovery testing",
            "Service the press seals and switch to food-grade lubricant",
            "Verify room classification and HEPA filter integrity",
        ),
        CRIT,
        "Foreign material in or on the tablet - a patient safety issue",
    ),
    ("pill", "crack"): DefectKnowledge(
        "compression",
        (
            "Excessive compression force exceeding the granule binding capacity",
            "Insufficient binder in the granulation",
            "Capping from entrapped air during rapid compression",
        ),
        (
            "Reduce main compression force and re-verify tablet hardness",
            "Adjust the binder level and revalidate the granulation",
            "Add or extend pre-compression to allow air escape",
        ),
        HIGH,
        "Fracture in the tablet, affecting dose integrity",
    ),
    ("pill", "faulty_imprint"): DefectKnowledge(
        "compression / punch tooling",
        (
            "Punch tip wear degrading the embossed detail",
            "Material adhering to the punch face (picking)",
            "Insufficient compression force to form the imprint fully",
        ),
        (
            "Replace the punch set and record tooling cycle count",
            "Add anti-adherent to the formulation or polish the punch faces",
            "Increase compression force within the validated range",
        ),
        MED,
        "Imprint incomplete or illegible - an identification failure",
    ),
    ("pill", "pill_type"): DefectKnowledge(
        "line clearance / changeover",
        (
            "Inadequate line clearance leaving product from the prior batch",
            "Mixed product in a shared hopper or container",
            "Mislabelled intermediate container",
        ),
        (
            "Reinforce line-clearance procedure with independent verification and sign-off",
            "Implement dedicated or fully verified single-product hoppers",
            "Add barcode verification at every container transfer",
        ),
        CRIT,
        "Wrong product present in the batch - a critical mix-up",
    ),
    ("pill", "scratch"): DefectKnowledge(
        "coating / handling",
        (
            "Tablet-to-tablet abrasion from excessive coating pan speed",
            "Abrasion in the conveying or deduster path",
            "Over-long tumbling dwell in the coating pan",
        ),
        (
            "Reduce pan rotation speed and increase the baffle count",
            "Add low-friction liners to the conveying path",
            "Shorten the tumbling dwell within the validated window",
        ),
        LOW,
        "Surface abrasion on the tablet or its coating",
    ),
    # ----------------------------------------------------------------- screw
    ("screw", "manipulated_front"): DefectKnowledge(
        "cold forming / heading",
        (
            "Heading die wear producing an out-of-spec drive recess",
            "Punch misalignment deforming the head profile",
            "Material flow defect from inconsistent wire feedstock",
        ),
        (
            "Replace the heading die and reset the cycle counter",
            "Realign the punch and verify concentricity",
            "Tighten incoming wire specification and supplier controls",
        ),
        HIGH,
        "Head or drive recess geometry deviating from specification",
    ),
    ("screw", "scratch_head"): DefectKnowledge(
        "handling / driving",
        (
            "Driver bit slip during a torque test (cam-out)",
            "Part-to-part abrasion in bulk handling",
            "Contact with the sorting bowl track",
        ),
        (
            "Replace worn driver bits and verify the torque profile",
            "Reduce bulk container fill depth to limit part-on-part load",
            "Line the sorting bowl track with a low-friction coating",
        ),
        LOW,
        "Surface scoring on the screw head",
    ),
    ("screw", "scratch_neck"): DefectKnowledge(
        "forming / handling",
        (
            "Abrasion against forming tooling at the head-to-shank transition",
            "Contact damage in the plating barrel",
            "Conveyor track wear scoring the neck radius",
        ),
        (
            "Polish the forming tooling at the transition radius",
            "Reduce plating barrel load and rotation speed",
            "Replace the worn conveyor track section",
        ),
        LOW,
        "Scoring in the neck region below the head",
    ),
    ("screw", "thread_side"): DefectKnowledge(
        "thread rolling",
        (
            "Thread rolling die wear producing an incomplete flank",
            "Die misalignment creating asymmetric thread form",
            "Insufficient rolling pressure to fully form the thread",
        ),
        (
            "Replace the thread rolling dies and log the cycle count",
            "Realign the die set and verify thread form with a profile gauge",
            "Increase rolling pressure to the validated setpoint",
        ),
        HIGH,
        "Thread flank malformed on the side profile - assembly risk",
    ),
    ("screw", "thread_top"): DefectKnowledge(
        "thread rolling",
        (
            "Die crest wear flattening the thread peak",
            "Blank diameter below tolerance leaving the crest unfilled",
            "Rolling pressure insufficient to form the full profile",
        ),
        (
            "Replace the rolling dies and verify crest geometry",
            "Tighten incoming blank diameter inspection",
            "Increase rolling pressure and re-verify with a thread gauge",
        ),
        HIGH,
        "Thread crest malformed or incompletely formed",
    ),
    # ------------------------------------------------------------------ tile
    ("tile", "crack"): DefectKnowledge(
        "firing / cooling",
        (
            "Thermal shock from an excessive kiln cooling rate",
            "Body moisture above specification entering the kiln",
            "Mechanical stress during unloading and stacking",
        ),
        (
            "Reduce the kiln cooling ramp and verify the firing curve",
            "Extend pre-drying and add a moisture check before the kiln",
            "Add cushioned handling at the unloading station",
        ),
        CRIT,
        "Structural fracture through the tile body",
    ),
    ("tile", "glue_strip"): DefectKnowledge(
        "handling / packaging",
        (
            "Adhesive residue from packaging tape or spacers",
            "Conveyor belt adhesive transfer onto the tile face",
            "Label adhesive migrating onto the glazed surface",
        ),
        (
            "Qualify a low-residue packaging tape",
            "Clean or replace the conveyor belt and add it to the PM schedule",
            "Switch to a removable label adhesive",
        ),
        LOW,
        "Adhesive residue in a strip pattern on the surface",
    ),
    ("tile", "gray_stroke"): DefectKnowledge(
        "glazing / printing",
        (
            "Glaze application streaking from a partially blocked spray nozzle",
            "Print roller wear producing an inconsistent decoration pass",
            "Glaze viscosity drift altering flow behaviour",
        ),
        (
            "Clean or replace the spray nozzle and verify the spray pattern",
            "Replace the print roller and log the cycle count",
            "Add continuous glaze viscosity monitoring with correction",
        ),
        MED,
        "Streak or stroke mark in the glaze or decoration",
    ),
    ("tile", "oil"): DefectKnowledge(
        "pressing / conveying",
        (
            "Hydraulic press oil leak contaminating the tile body",
            "Conveyor chain lubricant over-application dripping onto product",
            "Compressed-air line carrying oil mist onto the surface",
        ),
        (
            "Repair the press hydraulic seal and add leak detection",
            "Reduce chain lubrication volume and add drip containment",
            "Install a coalescing filter on the compressed-air supply",
        ),
        HIGH,
        "Oil contamination staining the tile, preventing glaze adhesion",
    ),
    ("tile", "rough"): DefectKnowledge(
        "glazing / firing",
        (
            "Glaze application too thin to level into a smooth surface",
            "Firing temperature below the glaze maturing range",
            "Particulate settling on the glaze before firing",
        ),
        (
            "Increase glaze application weight to the validated range",
            "Recalibrate the kiln profile and verify with witness cones",
            "Improve pre-kiln air filtration and add a blow-off station",
        ),
        MED,
        "Surface texture rougher than the finish specification",
    ),
    # ------------------------------------------------------------ toothbrush
    ("toothbrush", "defective"): DefectKnowledge(
        "injection moulding / bristle tufting",
        (
            "Bristle tufting anchor failure leaving deformed or missing tufts",
            "Injection moulding parameter drift producing a short shot or flash",
            "Trim station misalignment leaving an uneven bristle profile",
        ),
        (
            "Recalibrate the tufting anchor force and verify tuft retention by pull test",
            "Reset the moulding parameters and verify with a first-article check",
            "Realign the bristle trim station and verify the profile height",
        ),
        MED,
        "Bristle field or handle geometry deviating from specification",
    ),
    # ------------------------------------------------------------ transistor
    ("transistor", "bent_lead"): DefectKnowledge(
        "lead forming / handling",
        (
            "Lead-forming die misalignment bending the lead out of plane",
            "Handling damage in the transport tray or tube",
            "Pick-and-place nozzle contacting the lead during placement",
        ),
        (
            "Realign the lead-forming die and verify with a coplanarity check",
            "Switch to a tray with individual lead protection pockets",
            "Recalibrate the pick-and-place nozzle offset",
        ),
        HIGH,
        "Lead deformed out of the coplanarity specification",
    ),
    ("transistor", "cut_lead"): DefectKnowledge(
        "lead trimming",
        (
            "Trim die wear producing an incomplete or ragged cut",
            "Trim length set outside the specification",
            "Trim blade misalignment cutting into the lead body",
        ),
        (
            "Replace the trim die and reset the cycle counter",
            "Recalibrate trim length and add first-article verification",
            "Realign the trim blade and verify the cut position",
        ),
        CRIT,
        "Lead severed or trimmed short - the device cannot be assembled",
    ),
    ("transistor", "damaged_case"): DefectKnowledge(
        "moulding / handling",
        (
            "Ejection damage as the part is pushed from the mould",
            "Impact during handling or transport",
            "Excessive clamping force at a downstream test station",
        ),
        (
            "Adjust ejector pin timing and balance the ejection force",
            "Add cushioning to transport trays and reduce drop heights",
            "Reduce test-station clamping force and add force monitoring",
        ),
        HIGH,
        "Package body cracked or deformed, risking die exposure",
    ),
    ("transistor", "misplaced"): DefectKnowledge(
        "pick and place",
        (
            "Placement head calibration drift shifting the part position",
            "Vision alignment failure from poor fiducial contrast",
            "Component sliding on the solder paste before reflow",
        ),
        (
            "Recalibrate the placement head and verify with a placement accuracy test",
            "Improve fiducial lighting and revalidate the vision routine",
            "Adjust the solder paste tack and verify the print deposit",
        ),
        CRIT,
        "Component placed outside its positional tolerance",
    ),
    # ------------------------------------------------------------------ wood
    ("wood", "color"): DefectKnowledge(
        "drying / finishing",
        (
            "Kiln drying temperature variation producing uneven discoloration",
            "Natural grain and heartwood variation not sorted before finishing",
            "Stain application inconsistency across the board width",
        ),
        (
            "Improve kiln airflow uniformity and verify with a load survey",
            "Add a colour-sort step before finishing",
            "Recalibrate the stain applicator and verify coverage uniformity",
        ),
        LOW,
        "Colour variation beyond the appearance grade specification",
    ),
    ("wood", "combined"): DefectKnowledge(
        "milling / finishing",
        (
            "Several process parameters drifting simultaneously",
            "A raw material batch problem manifesting in multiple ways",
            "Deferred maintenance allowing multiple faults to accumulate",
        ),
        (
            "Stop the line and audit the full process before resuming",
            "Quarantine and evaluate the raw material batch",
            "Bring the maintenance schedule current and verify each subsystem",
        ),
        HIGH,
        "Multiple defect modes present together",
    ),
    ("wood", "hole"): DefectKnowledge(
        "raw material / milling",
        (
            "Natural knot dislodging during machining",
            "Insect damage present in the raw timber",
            "Drill or router breakthrough beyond the intended depth",
        ),
        (
            "Tighten raw-material grading to exclude loose knots",
            "Review timber storage and treatment for pest control",
            "Reset the machining depth and verify with a first-article check",
        ),
        MED,
        "Void or perforation in the wood surface",
    ),
    ("wood", "liquid"): DefectKnowledge(
        "finishing / handling",
        (
            "Resin bleed from insufficiently dried heartwood",
            "Machine lubricant dripping onto the workpiece",
            "Finish application pooling from over-application",
        ),
        (
            "Extend the kiln schedule to fully set the resin",
            "Repair the lubricant leak and add drip containment",
            "Recalibrate the finish applicator flow rate",
        ),
        MED,
        "Liquid stain or residue on the surface",
    ),
    ("wood", "scratch"): DefectKnowledge(
        "milling / handling",
        (
            "Debris trapped under a feed roller scoring the surface",
            "Worn or nicked planer knife leaving a repeating mark",
            "Board-to-board abrasion during stacking",
        ),
        (
            "Add a blow-off station ahead of the feed rollers",
            "Replace the planer knives and inspect for nicks each shift",
            "Introduce interleaving separators when stacking",
        ),
        LOW,
        "Linear surface scoring",
    ),
    # ---------------------------------------------------------------- zipper
    ("zipper", "broken_teeth"): DefectKnowledge(
        "teeth forming / attachment",
        (
            "Insufficient crimping force leaving teeth poorly retained",
            "Material embrittlement from over-aggressive forming",
            "Impact damage during handling and spooling",
        ),
        (
            "Recalibrate the crimping force and add pull testing",
            "Reduce the forming rate and verify material ductility",
            "Reduce spooling tension and add cushioned guides",
        ),
        HIGH,
        "One or more teeth fractured or missing - the zipper will fail",
    ),
    ("zipper", "combined"): DefectKnowledge(
        "assembly",
        (
            "Multiple parameters out of control simultaneously",
            "An upstream tape defect cascading into several symptoms",
            "Machine fault ignored rather than escalated",
        ),
        (
            "Stop the line and audit the process before resuming production",
            "Trace the tape supply and quarantine the affected roll",
            "Reinforce the fault-escalation procedure",
        ),
        CRIT,
        "Several defect modes co-occurring on one zipper",
    ),
    ("zipper", "fabric_border"): DefectKnowledge(
        "tape weaving / edge finishing",
        (
            "Edge fraying from a dull or worn cutting blade",
            "Insufficient heat sealing at the tape edge",
            "Weaving tension variation distorting the border",
        ),
        (
            "Replace the cutting blade and add it to the PM schedule",
            "Increase the edge-seal temperature within the validated range",
            "Restore weaving tension control and verify the load cell",
        ),
        MED,
        "Tape border frayed, distorted, or unsealed",
    ),
    ("zipper", "fabric_interior"): DefectKnowledge(
        "tape weaving",
        (
            "Weft thread break leaving a flaw in the tape body",
            "Loom timing error producing an irregular weave",
            "Yarn quality variation from the supplier",
        ),
        (
            "Improve yarn tension control to reduce break frequency",
            "Re-time the loom and verify the weave pattern",
            "Tighten incoming yarn specification and supplier monitoring",
        ),
        MED,
        "Weave defect within the tape body",
    ),
    ("zipper", "rough"): DefectKnowledge(
        "finishing",
        (
            "Incomplete deburring after teeth forming",
            "Insufficient lubrication leaving a coarse running surface",
            "Surface treatment applied unevenly",
        ),
        (
            "Add or extend the deburring step and verify by touch inspection",
            "Recalibrate the lubricant application rate",
            "Verify surface treatment coverage across the full run",
        ),
        LOW,
        "Rough surface texture affecting slider action",
    ),
    ("zipper", "split_teeth"): DefectKnowledge(
        "teeth forming",
        (
            "Forming die wear producing an incomplete tooth profile",
            "Material flow defect from inconsistent wire feedstock",
            "Excessive forming force splitting the tooth",
        ),
        (
            "Replace the forming die and log the cycle count",
            "Tighten incoming wire specification and supplier controls",
            "Reduce forming force to the validated setpoint",
        ),
        HIGH,
        "Tooth split along its profile, compromising engagement",
    ),
    ("zipper", "squeezed_teeth"): DefectKnowledge(
        "teeth forming / attachment",
        (
            "Excessive crimping force deforming the tooth geometry",
            "Die clearance below specification over-compressing the tooth",
            "Misalignment between the tape and the crimping station",
        ),
        (
            "Reduce crimping force and verify tooth geometry with a gauge",
            "Reset the die clearance to the validated dimension",
            "Realign the tape guide into the crimping station",
        ),
        HIGH,
        "Tooth compressed out of profile, preventing proper engagement",
    ),
}


def get_knowledge(category: str, defect_type: str) -> DefectKnowledge | None:
    """Look up what is known about one defect mode."""
    return KNOWLEDGE_BASE.get((category, defect_type))


def known_defect_types(category: str) -> list[str]:
    return sorted(d for (c, d) in KNOWLEDGE_BASE if c == category)


def known_categories() -> list[str]:
    return sorted({c for (c, _) in KNOWLEDGE_BASE})
