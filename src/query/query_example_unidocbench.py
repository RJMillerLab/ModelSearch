
TEMPLATES_ENG_SCI = """
**Factual Retrieval:**
    1. What are the key components, inputs, or modules used in [System/Design]?
        - Example: What are the key propulsion components used in the Falcon 9 launch system?
    2. Which standards or protocols govern [Process/Device] in [Field]?
        - Example: Which IEEE standards apply to Wi-Fi signal modulation in telecom systems?
    3. What are the parameters or performance characteristics of [Device/Model]?
        - Example: What are the voltage and power ratings for the STM32F4 microcontroller?
    4. What experimental setup or simulation configuration is described in [Document/Study]?
        - Example: What experimental setup was used to measure aerodynamic drag in the wind tunnel tests?
    5. What equations, models, or formulas are used to analyze [Phenomenon]?
        - Example: What equations are used to model thermal expansion in metal rods?
    6. What are the failure modes or safety limits noted for [Component/Material]?
        - Example: What are the fatigue limits of 7075-T6 aluminum alloy under cyclic loading?
    7. When was [Instrument/System/Tool] calibrated, tested, or deployed?
        - Example: When was the onboard navigation system on the satellite last tested?
    8. Who developed or validated [Algorithm/Model/Tool]?
        - Example: Who developed the Kalman filter algorithm used in aircraft tracking?
    9. How is [Process/Concept] defined or implemented in [Field or System]?
        - Example: How is entropy defined and calculated in statistical mechanics?
    10. How do you assemble, operate, or troubleshoot [Machine/System/Experiment]?
        - Example: How do you calibrate and operate a laser Doppler vibrometer?
    11. What steps are involved in designing or verifying [System/Component]?
        - Example: What steps are involved in verifying the structural integrity of a rocket payload bay?

**Comparison:**
    1. Which [Design/Model/Method] performs better under [Condition/Use Case]?
        - Example: Which antenna design performs better for low-Earth orbit communications: patch or helical?
    2. How do the properties of [Material/Component] differ between [Version A] and [Version B]?
        - Example: How do the tensile strengths of carbon fiber composites differ from aluminum alloys?
    3. What are the differences between [Theory A] and [Theory B] in explaining [Phenomenon]?
        - Example: What are the differences between Newtonian and relativistic mechanics in orbital calculations?
    4. How does performance change when [Parameter/Condition] is altered in [System]?
        - Example: How does signal-to-noise ratio change with bandwidth in a 5G communication link?
    5. Which modeling approach gives more accurate results for [Problem]?
        - Example: Which method gives more accurate results in fluid simulation: finite element or lattice Boltzmann?
    6. How do implementations of [Algorithm/Model] vary across [Platforms/Fields]?
        - Example: How does the implementation of FFT differ between digital signal processors and GPUs?
    7. Which statistical method is more appropriate for analyzing [Data Type/Experiment]?
        - Example: Which statistical test is more suitable for evaluating variance in machine failure rates?

**Summarization:**
    1. What are the main design principles or system behaviors described in [Document/Prototype]?
        - Example: What are the main design principles behind the reusable thermal protection system in the Dragon capsule?
    2. Summarize the results of [Experiment/Simulation] in terms of [Metric/Effect].
        - Example: Summarize the results of the CFD simulation in terms of pressure drop across the turbine blade.
    3. How did [Variable] affect [Outcome] in the analysis or test?
        - Example: How did increasing angular velocity affect stress distribution in the rotating shaft?
    4. List the most significant performance trade-offs or limitations mentioned in [Design/System].
        - Example: What are the major trade-offs of using PID controllers in quadcopter stabilization?
    5. What challenges or anomalies were reported in [System/Test Case]?
        - Example: What anomalies were encountered during the vacuum chamber test of the cubesat propulsion unit?
    6. What alternative approaches or optimizations were proposed for [Problem]?
        - Example: What optimizations were suggested to improve heat dissipation in the processor array?
    7. What are the key theoretical insights, assumptions, or simplifications used in [Study/Model]?
        - Example: What assumptions are made in the derivation of the Black-Scholes equation?

**Causal / Reasoning / Why Questions:**
    1. Why did [System/Test/Model] produce unexpected or suboptimal results?
        - Example: Why did the RF signal degrade rapidly during the satellite communication test?
    2. How did [Material/Design Parameter] influence [Performance/Outcome]?
        - Example: How did adding graphite fiber influence the tensile strength of the composite?
    3. What was the motivation behind selecting [Approach/Architecture] for [Design/Project]?
        - Example: What was the motivation behind selecting distributed control in the Mars rover design?
    4. Why was [Technique/Tool/Algorithm] chosen over other alternatives in [Context]?
        - Example: Why was k-means clustering chosen for fault detection in smart grids?
    5. What were the implications of [Experimental Outcome/Failure] for future iterations or versions?
        - Example: What were the implications of the failed nozzle separation in the static fire test?
    6. In what sequence did key events or computations occur in [Process/Simulation]?
        - Example: In what sequence did the finite element solver execute stress-strain iterations during load application?
"""
