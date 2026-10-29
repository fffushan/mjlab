# X2 VAE decoder: literature on latent use and failure to start motion

Date: 2026-10-28 (+08).
Scope: literature review and interpretation of existing software and simulation evidence.
No new model evaluation, training, export, simulator run, or robot test was performed.

## 1. Answer

Yes. The literature discusses closely related failures, including a policy that never
starts moving from rest. It uses several different terms:

- **Posterior collapse:** the encoder posterior loses information by matching the prior.
- **Latent underuse:** changing the latent has little effect on the decoder output.
- **Copycat or inertia problem:** the policy predicts actions from previous actions instead
  of the information needed to change behavior.
- **Proprioceptive dominance:** robot-state inputs suppress learning from another input branch.
- **Distribution shift:** deployment uses state–command combinations absent from training.
- **History dependence:** a trajectory generator keeps extending the current behavior despite a new goal.

These terms do not identify the same failure. More than one can apply to one system.

The best current description of the X2 result is **weak latent control authority at the
measured standing state**. Complete posterior collapse is not established. Neither is an
architectural limit that restricts all VAE decoders to small pose corrections.

The decoder is a **state-conditioned feedback policy**. The latent should select the
short-term motion intent. The state inputs should implement that intent under current dynamics.
Training can instead produce a policy that mainly continues its current behavior.

## 2. Sources with the closest match

### 2.1 BeyondMimic itself: history can prevent behavior changes

Source [1], the user's local PDF, is arXiv:2508.08241v4, dated 13 November 2025.
The relevant section is **“Limtations and Future Works,” printed page 12**.

The paper says history can cause the model to become:

> “trapped in repetitive motion patterns even when guidance is active.”

It also says the robot:

> “tends to stumble at the start and end of motions.”

The authors report that stronger guidance can help escape repetition but can destabilize
denoising. They propose reducing history dependence as future work.

**Relevance:** this directly concerns motion initiation and changes between behaviors.
It is not a decoder-only diagnosis. The passage concerns the guided diffusion system.
It does not show that a decoder ignores all latent inputs at a standing state.

**What the decoder is intended to learn:** Methods, printed pages 19–21, separates a
reference encoder from a proprioception-conditioned action decoder. The encoder also
receives measured anchor error. The action decoder uses the latest observations at execution.
Thus, a latent is not necessarily a state-independent motion label.

The paper's stated motivation is to separate smooth motion representations from sharp,
feedback-dependent motor commands. Its cartwheel ablation reports 5% success without latent
encoding and 95% with latent encoding in MuJoCo, printed page 23. This is evidence for
that system and task, not an X2 comparison or a guarantee of latent controllability.

### 2.2 Motion VAEs: a section specifically about ignoring the latent

Ling et al., **Character Controllers Using Motion VAEs**, SIGGRAPH / ACM TOG 2020 [2].
Read **Section 3.3, “Avoiding Posterior Collapse.”**

The authors report that additional past-pose conditioning improves reconstruction but reduces
output diversity. They warn that:

> “the decoder may learn to ignore the encoder output”

Their practical measures include:

- Use only one or two past poses in their experiments.
- Feed the latent to every layer of each decoder expert.
- Consider decoder size, KL weight, and latent dimension together.
- Test whether the learned representation actually permits downstream control.

Their model encodes **pose transitions**, rather than treating each latent as an entire skill.
The subsection “Testing MVAE Generalization” includes a running-task control check because reconstruction alone does not establish
that the latent space supports useful control.

**Relevance:** this is a very close architectural explanation for the suspected shortcut.
**Limit:** this is kinematic character animation, not a torque-driven X2 policy.
The conditioning variables and physical safety requirements differ.

### 2.3 Copycat policies: an explicit report of never starting from rest

Wen et al., **Fighting Copycat Agents in Behavioral Cloning from Observation Histories**,
2020 [3]. Read **Sections 1 and 4**.

The paper gives a simple example. A car brakes at a red light and accelerates after green.
A policy that repeats the previous action predicts almost every demonstration sample correctly.
It misses the one decision that changes the behavior. During deployment, it can remain stopped.

Section 4 reports:

> “copycat Half-Cheetah agents that did not ever start to move from a resting position.”

The same section shows a Walker2D knee failing to change from an extended to a bent position.
The paper finds that lower held-out action-prediction loss can coexist with worse policy return.

**Relevance:** this is the closest observed symptom among the sources checked.
The X2 decoder receives `previous_action` directly. It does not need to infer that signal
from images, as some policies in this paper do.

**Limit:** the paper does not study the X2 VAE or latent diffusion. A high correlation between
successive X2 actions does not by itself prove a copycat policy. Smooth physical control also
requires correlated commands.

Related work:

- **Causal Confusion in Imitation Learning**, de Haan et al., NeurIPS 2019 [4], studies
  predictive shortcuts that fail under policy-induced changes in the observation distribution.
- **Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction**,
  Chuang et al., ECCV 2022 [5], explicitly calls this the copycat or inertia problem.
  Its remedy uses a separate history branch trained on action differences and controlled
  information flow. It is not equivalent to merely replacing absolute actions with deltas.

### 2.4 PULSE: the direct predecessor also reports prolonged standing

Luo et al., **Universal Humanoid Motion Representations for Physics-Based Control**,
ICLR 2024 [6]. BeyondMimic cites this work as reference 74 for its VAE design.

PULSE distills a motion imitator into a latent-conditioned policy. It also learns a
**state-conditioned prior**, rather than using only a fixed standard-normal prior.
Section 4.2 explicitly contrasts the action distributions for standing and flipping in midair.

The authors' project page, **Random Motion Generation → Random Sampling**, states:

> “the humanoid can sometimes stand still for a long time before starting to move again.”

This occurs with small sampling variance. Larger variance produces more energetic motion,
but also more falls. The page also demonstrates recovery and switching between imitation
and random generation.

**Relevance:** prolonged standing is a documented behavior in the immediate design lineage.
Latent sampling and current state both matter.
**Limit:** this is an author-reported random-sampling observation, not proof that commanded
motion fails or that the decoder ignores the latent. PULSE uses a simulated human model.

PULSE's encoder inputs, learned prior, motion repertoire, and recovery training differ from X2.
Its results cannot justify increasing the X2 latent magnitude without a bounded simulation test.

### 2.5 Physics-based character control explicitly discusses posterior collapse

Bae et al., **Versatile Physics-based Character Control with Hybrid Latent Representation**,
Eurographics / Computer Graphics Forum 2025 [7]. Read **Sections 2.2, 3.2, and 4.1–4.2**.

The paper discusses continuous VAE priors that lose useful latent information. It also
reports difficulties keeping downstream latent search inside the valid motion region.
Its proposed representation combines discrete quantization with continuous residuals.

**Relevance:** latent collapse is discussed inside physics-based character control, not only
in image or language VAEs. The paper treats representation quality and downstream control
as separate concerns.
**Limit:** quantization is an alternative design, not a demonstrated repair for the X2 failure.
It does not remove the need for state coverage, transition data, or closed-loop tests.

### 2.6 Proprioception can suppress the input that should change the motion

Lu et al., **When would Vision-Proprioception Policies Fail in Robotic Manipulation?**,
ICLR 2026 [8].

The authors examine phases within a trajectory. They find that concise proprioceptive inputs
can reduce training loss faster and suppress learning from vision during motion-transition phases.
Their Gradient Adjustment with Phase-guidance method changes optimization of the proprioceptive branch.

**Relevance:** the competing input in X2 is a reference-derived latent, not vision. Still,
this provides a tested mechanism for one informative input branch to dominate another.
The important control decision may occupy a small fraction of training samples.
**Limit:** manipulation results do not establish that this mechanism caused the X2 result.

## 3. What these decoders are meant to learn

### 3.1 An inverse-control mapping, not a pose renderer

Merel et al., **Neural Probabilistic Motor Primitives for Humanoid Control**, ICLR 2019 [9],
provides a useful interpretation. Section 2.2 describes an inverse model with a latent-variable
bottleneck. A short future-state snippet supplies intent. The decoder combines its latent
with the current state to choose an action.

Using simplified notation:

```text
z_t ~ encoder(reference or future snippet, relevant state error)
a_t = decoder(current proprioception, previous_action, z_t)
```

The motion results from repeated policy execution and the physical dynamics. The latent is
usually a **per-step control representation**, not a complete motion clip or a named skill.
A running motion can require an evolving sequence of latents.

Merel et al. also study state perturbations and local feedback-policy cloning.
Their experts are nonlinear feedback controllers around reference trajectories.
This matters because nominal trajectory samples alone do not describe how to act away from them.

### 3.2 Why a useful-looking model can learn only small corrections

For illustration, consider the distillation loss:

```text
L = E[||D(c, z) - a_teacher||^2] + beta * KL(q(z | r) || p(z))
```

Here `c` is decoder conditioning and `r` is encoder input. This notation is illustrative;
exact encoder conditioning and loss reductions differ between papers and implementations.

Suppose the training trajectories are smooth. Current joint state and previous action already
predict most of the next teacher action. A model can obtain low error with:

```text
D(c, z) ≈ continuation(c) + small_intent_correction(c, z)
```

If the correction gives little average reconstruction benefit, the KL penalty discourages
spending latent information on it. The model need not learn a state-independent command channel.
The important question is what the latent adds **after conditioning is known**.

This is a plausible explanation, not a measured decomposition of the X2 network.
In particular, a small correction can still be essential for sustained motion or phase changes.

### 3.3 A latent valid during fast motion may not be valid at rest

Training constrains the decoder on joint samples `(conditioning, latent)`. It does not constrain
all combinations of independently sampled conditioning and latent.

A latent from mid-swing can be valid with mid-swing joint velocities and previous actions.
The same latent paired with zero velocity and standing action history can be out of distribution.
Both inputs can be individually common while their combination is absent from training.

This is one reason PULSE and **ControlVAE** [10] learn state-conditioned priors.
ControlVAE evaluates prior variants and reports stopping, starting, turning, and other behaviors
under state-conditioned random sampling. Thus, state-conditioned VAE policies are not inherently
restricted to small pose adjustments. ControlVAE uses model-based learning, not the same
teacher-distillation objective as BeyondMimic.

A conditional prior alone does not guarantee responsiveness. It can still concentrate near a
standing continuation if transitions are rare or its variance is small.

### 3.4 Direct studies of whether a robotic CVAE uses its latent

Kang, **The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action
Chunking Transformers**, 2026 preprint [11], measures latent use directly.

On the tested ACT manipulation benchmarks, posterior sampling gives little reconstruction
benefit at the tested positive KL weights. Replacing or shuffling latents changes little.
The paper includes positive controls where latent information does help.
It also reports that activating latent dimensions with free bits need not improve reconstruction.

**Relevance:** successful policy execution and a VAE-shaped architecture do not establish
that the latent carries useful intent. Measure its contribution directly.
**Limit:** this is a recent preprint with benchmark-specific results and limited seeds for
some probes. ACT normally uses `z=0` at inference. X2 instead depends on a generated latent.
The two deployment roles are materially different.

## 4. Interpretation of the X2 findings

Input: [the 2026-10-27 findings note](x2_vae_latent_decoder_findings_20261027.md).
The measurements below are reported there; this review did not rerun them.

| Existing observation | Supported reading | What remains unproved |
|---|---|---|
| Checkpoint, ONNX, and recorded actions agree closely | The tested export is faithful | Policy quality or correct planner semantics |
| Latent swaps change standing actions much less than moving-state actions | Latent authority depends on the conditioning state | No feasible latent sequence can initiate motion |
| Standing Jacobian effective rank is about 2 | Local control variation is concentrated | Global posterior collapse or only two useful directions everywhere |
| Previous action has strong level attribution | A continuation shortcut is plausible | Previous action is unnecessary or harmful in every regime |
| Latent has the largest reported increment share, 0.462 | The latent is not simply absent from the tested computation | Increment attribution alone establishes causal task control |
| A live encoder sequence tracks the tested fast reference better than fixed latents | Temporal latent content and/or live state-error encoding matters | This same policy can start that reference from an arbitrary standing state |
| The diffusion store has no measured cold starts | Diffusion startup coverage is a concern | The separate VAE DAgger dataset lacks the same samples |

### 4.1 Conclusions that need narrower wording

1. **A fixed latent fixes the policy, not the physical state.** An autonomous feedback system
   can have equilibria, limit cycles, or unstable trajectories. A fixed latent does not
   mathematically require standing or one unique equilibrium. The reported five tests establish
   behavior for those latents and initial states only.
2. **A fixed feedback law can correct an offset it did not create.** This is a basic role of
   feedback. Failure to recover from one offset does not establish the opposite general rule.
3. **The 1/85 ratio is not a required control-authority threshold.** Its denominator compares
   actions from different physical regimes. The action needed to initiate motion from standing
   must be established at that same standing state.
4. **Scaling one latent direction is not a search over all latents.** The reported tenfold
   scaling result supports limited authority on that tested direction. It does not establish
   global saturation or impossibility of motion initiation.
5. **Low latent rank is not by itself posterior collapse.** A low-dimensional useful control
   representation can be intentional. Inspect posterior means, variances, KL, and intervention effects.
6. **A held plan array is not a constant executed latent.** Advancing through different rows
   changes the input even when the array does not change. The report gives a nonzero latent-step
   norm. Verify the selected latent sequence and its phase, rather than inferring constancy
   from the planner update rate.

These qualifications do not dismiss the standing failure. They prevent choosing a repair from
a stronger causal claim than the evidence supports.

### 4.2 Operator clarification: standing-start controls already pass

Reported by the operator on 2026-10-28 (+08). These are existing observations, not new tests
performed for this review. No new trace or trial count was supplied.

All three paths start under a standing policy that holds the robot still. The operator then
switches to the tennis teacher, full VAE, or diffusion policy. Thus, successful tracking does
not depend on initializing the robot inside an already moving reference trajectory.

| Path | Operator-reported result |
|---|---|
| Frozen teacher + live reference | Starts from standing and tracks the reference without difficulty |
| Full VAE + live encoder | Starts from standing and tracks at nearly teacher performance |
| Diffusion + 36-teacher VAE decoder | Remains standing, although the latent changes from nearer standing to nearer moving |
| Diffusion + 8-teacher VAE decoder, simulation | Stumbles initially, then performs some tennis motion |
| Diffusion + 8-teacher VAE decoder, physical robot | Does not recover from the initial stumble and falls immediately |

**Updated interpretation:** a usable latent sequence for the standing-to-tennis transition exists
through the live encoder. The decoder is not inherently limited to holding the current pose.
The failure appears when the live encoder path is replaced by the diffusion path, including
its conditioning, latent generation, timing, and handover behavior. This does not isolate the
diffusion network weights from the runtime, or establish decoder robustness to generated errors.

The 36-teacher result is a failure to leave standing, not necessarily a failure to follow a
command: the operator later clarified that the comparison concerns unguided diffusion.
The 8-teacher result initiates motion but has an unsafe startup transient in the reported
physical test. These are different observed outcomes, not proof that adding standing teachers
caused the difference. Latent proximity to a moving-data cluster does not establish the correct
action at the measured standing state. Exact action effects, state–latent consistency, and
temporal order matter.

Complete latent neglect is no longer the leading explanation for the end-to-end failure.
Weak off-distribution authority or a narrow valid latent region can still contribute.
The reported physical failure also shows that later simulation motion is not a hardware
startup acceptance result.

### 4.3 Operator correction: unguided diffusion has no reference-frame oracle

The operator identified a missing assumption in the proposed shadow comparison: unguided
36-motion diffusion has no specified clip or frame. A successful reference-conditioned VAE
and an unguided generator do not receive the same motion objective. Selecting an arbitrary
tennis reference would compare different intentions, not measure diffusion action error.

The same-state live-encoder comparison is valid only for a deliberately specified reference
condition. Even then, the encoder supplies one successful solution, not necessarily a unique
correct action. It is not the primary oracle for an unguided rollout.

Standing is part of the 36-motion repertoire. An unguided sample that predicts standing and
executes standing can be valid. Excessive standing may indicate a diversity or conditioning
problem, but this requires repeated-sample and suitable conditional-data comparisons.
A sample that predicts movement but executes standing is a different issue: state–latent
inconsistency or a mismatch between the planned and executed control paths.

## 5. Highest-value separating tests

These are proposals. Execution and training require separate authorization.

### 5.1 For unguided diffusion, compare predicted states with latent-driven execution

The operator has already supplied the teacher/full-VAE/diffusion standing-start observations;
see Section 4.2. No reference identity needs to be assigned to an unguided sample.

Use the generated state–latent plan itself as a testable prediction. In bounded simulation:

1. Save the complete simulator and relevant controller state at the plan's conditioning time.
   Preserve actual previous action, observation history, and actuator/internal state where relevant.
2. Generate one plan of paired states and latents. Keep the original frame and timing metadata.
3. Restore the saved state and execute that plan's applicable latent sequence at the correct
   timestamps, without replanning during this diagnostic horizon. Use fresh measured decoder
   conditioning at every tick. Never feed generated future states as if they were measurements.
4. Project the resulting simulated states into the same physical representation and coordinate
   frame as the plan. Undo normalization/projection as needed. Compare matching timestamps,
   respecting whether each state is before or after its associated action.
5. Report predicted versus achieved motion, startup transient, and plan-execution error. End the
   bounded trial on its declared termination conditions. Repeat on saved plans/seeds only within
   an explicitly authorized budget.

Interpretation:

- Predicted standing and achieved standing: the sample is internally consistent. This alone
  does not establish collapse or a failure of the unguided generative objective.
- Predicted motion but achieved standing: the generated state–latent pair does not produce its
  predicted motion under this decoder/environment path, or the timing/input contract differs.
- Predicted plausible motion but achieved stumble: inspect the first state/action divergence;
  the sample's physical executability is not established by its predicted poses.
- Fixed-plan execution produces motion, but live replanning does not: investigate replanning,
  history, latency, and row selection. The protocols differ, so this is diagnostic evidence,
  not automatic proof of a particular scheduler defect.

This test isolates a plan's execution consistency, not intended-reference tracking. A failing
result does not by itself locate the cause in the generator versus decoder versus runtime.
Generated states need not be exact physical forecasts, so assess systematic errors and multiple
samples rather than demanding exact numerical identity.

A same-state encoder shadow comparison remains an optional, reference-conditioned experiment.
Specify the clip and phase in advance, give the generator a compatible supported condition, and
recompute encoder state-error inputs from the same measured snapshot. The shadow branch has no
actuation and receives the active path's actual previous action. Do not choose a nearest latent
cluster after the fact and call its reference the ground truth for an unguided sample.

### 5.2 Measure conditional latent value on the actual distillation data

Compare a trained conditioning-only baseline against the full VAE on episode-held-out data.
Within the fixed VAE, compare true, zero, and suitably matched shuffled latents.
Report total action error, action-increment error, and errors near transition starts separately.
Use state-matched alternatives where possible; arbitrary shuffles can create invalid pairs.

Inspect the actual VAE DAgger data. The later diffusion rollout store is not a substitute.
Check reset states, reference phases, executed previous actions, and transition sample weights.
DAgger only covers the states and reference conditions that its collection process visits [12].

### 5.3 Separate information loss from control sensitivity

Measure:

- Per-dimension posterior means, posterior variances, and KL contributions.
- True-latent benefit relative to the same decoder with a baseline latent.
- Singular values of the scaled latent-to-action Jacobian at several states.
- The effect of latent sequences over a finite horizon, not only one-step action changes.
- Teacher/student disagreement at the same standing state and intended next motion.

Kinoshita et al. [13] study decoder constraints that control posterior collapse.
Their theory supports inspecting decoder sensitivity, but it does not certify robot stability.
Do not demand full rank across all 32 latent dimensions of a 31-action output.

### 5.4 Choose a repair only after the separating test

| Finding | Candidate change | Important limit |
|---|---|---|
| Missing start/stop examples | Collect competent transition and recovery supervision; balance those windows | Teacher must handle those states; separate VAE and diffusion collection |
| Previous-action shortcut | Controlled history ablation; transition-weighted loss; separate residual-supervision branch | Do not remove torque/contact information blindly; delta output alone is not a cure |
| Latent branch underused | Test latent injection at more layers; effective KL sweeps; free bits or schedule changes | More active units do not guarantee useful control or safe actions |
| Invalid state–latent combinations | Train/evaluate state-conditioned latent models and consistent planned state–latent pairs | A plausible rendered state trajectory is not proof of executable latents |
| History locks diffusion into standing | Compare aligned receding-horizon execution and history-conditioning variants | Stronger guidance or less history can reduce stability |

## 6. Reading order and conclusion

Read these first:

1. Wen et al. [3], Section 4: the explicit failure to start moving.
2. Ling et al. [2], Section 3.3: the decoder ignoring its latent.
3. BeyondMimic [1], printed page 12 and pages 19–21: history dependence and decoder intent.
4. PULSE [6], Section 4.2 and the project sampling notes: state-conditioned latent semantics.
5. Bae et al. [7], Sections 3.2–4.2: a physics-based alternative to continuous-prior collapse.

**Conclusion, updated after the operator clarification:** the teacher and full VAE both start
from a standing-policy handover. The failure appears in the diffusion-fed path. Therefore,
“the decoder can only refine an existing moving pose” does not explain the observed controls.
The leading investigation is generated-latent usability at the measured state, including
history, normalization, timing, and decoder tolerance. For unguided diffusion there is no
reference-frame oracle. Compare the generated state's motion with latent-driven execution from
the same saved simulator state. Predicted and achieved standing must be distinguished from
predicted motion that the generated latents cannot execute. The specific cause remains unknown.

## 7. References and checked locations

[1] Liao et al. **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided
Diffusion.** Local PDF `~/Documents/papers/beyondmimic.pdf`, arXiv v4.
[Full text](https://arxiv.org/html/2508.08241v4).
Checked: limitations; Methods “LDMs for Trajectory Modeling”; latent ablation; supplement S3.

[2] Ling et al. **Character Controllers Using Motion VAEs.** ACM TOG 39(4), 2020.
[Full text](https://arxiv.org/html/2103.14274),
[publication](https://doi.org/10.1145/3386569.3392422).
Checked: Sections 3.1–3.4 and “Testing MVAE Generalization.” The arXiv upload is later than publication.

[3] Wen et al. **Fighting Copycat Agents in Behavioral Cloning from Observation Histories.** 2020.
[Full text](https://arxiv.org/html/2010.14876).
Checked: Sections 1, 4, 5, 6.3, and appendix “Test Loss Comparison.”

[4] de Haan, Jayaraman, Levine. **Causal Confusion in Imitation Learning.** NeurIPS 2019.
[Paper](https://arxiv.org/abs/1905.11979).
Checked: abstract and its causal-misidentification explanation.

[5] Chuang et al. **Resolving Copycat Problems in Visual Imitation Learning via Residual Action
Prediction.** ECCV 2022. [Full text](https://arxiv.org/html/2207.09705).
Checked: introduction and Section 4.2, including separate branch targets and stop-gradient.

[6] Luo et al. **Universal Humanoid Motion Representations for Physics-Based Control (PULSE).**
ICLR 2024. [Full text](https://arxiv.org/html/2310.04582),
[author project page](https://www.zhengyiluo.com/PULSE-Site/).
Checked: Sections 4.1–4.2, 5.4, and project “Random Sampling.”

[7] Bae et al. **Versatile Physics-based Character Control with Hybrid Latent Representation.**
Eurographics / Computer Graphics Forum 2025.
[Full text](https://arxiv.org/html/2503.12814),
[project](https://jinseokbae.github.io/hybrid_latent_prior/).
Checked: Sections 2.2, 3.2, and quantization discussion in Sections 4.1–4.2.

[8] Lu et al. **When would Vision-Proprioception Policies Fail in Robotic Manipulation?**
ICLR 2026. [Full text](https://arxiv.org/html/2602.12032),
[conference record](https://openreview.net/forum?id=2RIqqNqALN).
Checked: abstract, optimization explanation, and visual-feature probing discussion.

[9] Merel et al. **Neural Probabilistic Motor Primitives for Humanoid Control.** ICLR 2019.
[Full text](https://arxiv.org/html/1811.11711).
Checked: Sections 2.2–2.3 and temporal-prior ablation.

[10] Yao et al. **ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based
Characters.** ACM TOG 2022. [Full text](https://arxiv.org/html/2210.06063),
[project](https://heyuanyao-pku.github.io/Control-VAE/).
Checked: Sections 3–4, random sampling, and Section 5.6 prior comparison.
Do not confuse this paper with the different “ControlVAE: Controllable Variational Autoencoder.”

[11] Kang. **The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action
Chunking Transformers.** 2026 preprint, v2. [Full text](https://arxiv.org/html/2609.16745v2).
Checked: abstract, Sections II-A, III-D, VI, and VII. Not treated as established consensus.

[12] Ross, Gordon, Bagnell. **A Reduction of Imitation Learning and Structured Prediction to
No-Regret Online Learning.** AISTATS 2011.
[Proceedings](https://proceedings.mlr.press/v15/ross11a.html).
Checked: abstract and description of training on policy-induced state distributions.

[13] Kinoshita et al. **Controlling Posterior Collapse by an Inverse Lipschitz Constraint on the
Decoder Network.** ICML 2023.
[Proceedings](https://proceedings.mlr.press/v202/kinoshita23a.html).
Checked: abstract and stated theoretical scope; no reproduction of its experiments.

## 8. Review limits

This is a targeted search, not a systematic review. Primary papers and author project pages
support the main claims. Search snippets were used for discovery, not as the sole evidence
for the quoted passages. Exa and AnySearch supplied search results; some fetches needed a
second public URL. Not every cited method shares X2's embodiment, observations, or training objective.
No literature result establishes the cause of this particular X2 failure.

The report uses short technical sentences. Word-level ASD-STE100 dictionary compliance was not checked.
