# BeyondMimic diffusion D6 — minimal hardware qualification plan

Date: 2026-10-07 (+08).
Status: planning only; no hardware operation is authorized by this document.

## Purpose

D6 checks whether the selected D5 runtime can be connected to the AgiBot X2
for a small, controlled research reproduction trial.
D6 is not a production deployment certification.
It does not claim general stability, safety, or performance from one trial.

D5 must finish its essential software, timing, export and vendor-sim2sim gates
before D6 hardware work starts.
The operator must separately authorize every robot operation.

## Scope

D6 covers only these physical interfaces:

- measured state and frame semantics;
- communication and command target;
- 31-D action mapping;
- operator-owned handover and release;
- one small bounded tracking trial;
- capture and review of the required evidence.

D6 does not require a large test matrix, repeated random trials, a delay sweep,
or broad hardware robustness testing.
Those are later research work unless the result requires them.

## Essential gates

### 1. Hardware preflight

Before motion, verify on the target system:

- the selected named policy and versioned runtime bundle are present;
- the controller loads the expected 167/99/31 contract;
- joint names and action order match the target robot;
- measured frame names and sign conventions match the D5 contract;
- command limits and the operator-selected safe release are configured;
- the process can stop and release commands under operator control.

Use configuration checks and recorded messages where possible.
Do not start a tracking motion during this gate.

### 2. State and communication check

Record a short stationary or otherwise operator-approved sample.
Check only the values needed by the controller:

- state dimensions and finite values;
- frame and timestamp continuity;
- measured joint ordering;
- target command ordering;
- command acknowledgement or delivery evidence;
- process stop and release evidence.

This is an interface check.
It is not a state-estimator quality or physical stability result.

### 3. Controlled handover and release

With operator authorization, perform one controlled handover to the selected
research policy and one explicit release back to the approved hold/control path.
Record:

- policy identity;
- handover and release times;
- command and measured-state dimensions;
- whether the runtime was active;
- stop reason and cleanup state.

Do not infer physical safety from a software `READY` or `ACTIVE` state.
The operator owns the physical safe-hold decision.

### 4. One bounded tracking trial

Run one named motion or one short approved segment.
Use the selected D5 path only.
Record:

- exact artifact and configuration identities;
- input and command coverage;
- measured joint response;
- attitude or fall-stop telemetry available from the system;
- runtime warnings and planner replacements;
- stop reason and cleanup.

Stop on the pre-authorized operator or system condition.
Do not expand the trial after an unexpected result without new authorization.

## Minimal evidence

D6 requires one concise report containing:

- source, policy, runtime and configuration hashes;
- target and frame mapping;
- preflight result;
- state/communication sample result;
- handover and release record;
- bounded-trial record, if authorized and completed;
- explicit limitations and unresolved risks.

The report must separate:

- software contract evidence;
- communication evidence;
- descriptive physical response;
- stability or failure observations;
- readiness decisions.

A missing or malformed record is unknown, not a pass.

## Not required for initial D6

The following are deliberately not completion gates for this research trial:

- exhaustive unit-test matrices;
- repeated randomized hardware trials;
- delay or dropout sweeps;
- multiple guided and unguided hardware modes;
- p99 production service-level objectives;
- broad actuator identification;
- generalization across all motions;
- deployment certification.

If the single bounded trial exposes a concrete failure, add only the smallest
reproduction needed to distinguish the relevant hypotheses.

## Exit criteria

D6 initial hardware qualification is complete only when the operator-authorized
preflight, interface check and bounded trial record are complete, or when the
operator records a clear blocked/failed outcome with preserved evidence.

A failed or blocked D6 result is valid research evidence.
It does not authorize retries or hardware changes.

Hardware stability, wider motion coverage, recovery behavior and deployment
acceptance remain later work unless the user defines a new scope.
