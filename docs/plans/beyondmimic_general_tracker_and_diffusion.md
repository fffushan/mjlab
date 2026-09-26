# General trackers, multimodality, and latent diffusion

Date: 2026-09-26.
Status: discussion and architectural rationale, not authorization for new training
or implementation. **Use the existing specialist teachers for the current
BeyondMimic distillation work.** A general teacher remains a future option.

Related documents:

- [VAE distillation architecture](beyondmimic_vae_distillation.md)
- [Implementation milestones](beyondmimic_vae_implementation.md)
- [M2 latent-core implementation and acceptance](beyondmimic_vae_m2_implementation.md)

## 1. The question

Is it worth training one general tracker across many reference trajectories? Can
multi-motion PPO work without an explicitly multimodal policy? If it can, why
train latent diffusion afterward, beyond its support for test-time guidance?

The motivating observation comes from autonomous driving: many different motions
can be reasonable in the same scenario. Averaging incompatible actions or
trajectories can produce an infeasible result. Passing an obstacle on either side
is valid; averaging the two paths may drive into it.

**Central distinction: a general tracker executes a specified motion; a planner
or generative model chooses which motion should happen next.**

## 2. Reference conditioning resolves much of the ambiguity

A general reference-conditioned tracker has the form

\[
a_t = \pi_\theta(s_t, r_{t:t+H}),
\]

where the reference or reference window communicates the intended motion. Here,
\(s_t\) denotes the available robot-state information, not necessarily a fully
observed physical state; \(H=0\) is possible for a current-reference-only tracker.

The difference is analogous to:

- **Planning:** get past the obstacle; choose left or right.
- **Tracking:** follow this particular left-passing trajectory.

For tennis, the same robot state can produce different actions when conditioned
on forehand versus backhand references. That is not contradictory supervision:
the inputs identify different tasks. These are illustrative motion choices, not
claims about the contents of the two selected tennis clips.

Even with a fixed reference, several stabilizing actions may be valid. Usually,
a tracker only needs one reliable solution for that condition, not a distribution
over every possible control strategy.

This makes broad tracking without diffusion possible. PHC demonstrates tracking
across ten thousand motion clips in simulation, using progressive policy
capacity. It is evidence that the problem is solvable without diffusion, not that
naive batching into a fixed small MLP will suffice, or that equivalent performance
on the physical X2 is established.

## 3. Mode averaging is not the same as mode collapse

Two failure modes should be separated:

- **Mode averaging:** the output lies between incompatible solutions and may be
  invalid.
- **Mode collapse or dropping:** the model retains one valid solution but loses
  other valid alternatives.

Supervised deterministic action regression with MSE, given ambiguous inputs,
learns the conditional mean. This directly creates the averaging problem.

PPO instead optimizes expected return. If two behaviors have high return and their
midpoint has low return, the objective does not prefer the midpoint merely because
two good behaviors exist. A simple policy may learn just one good behavior. That
may lack diversity, but is not necessarily infeasible. Optimization can still
fail, and a unimodal Gaussian does not faithfully represent arbitrary multimodal
action distributions.

**Averaging returns across correctly conditioned tasks is not the same as
averaging their actions.** Shared parameters can still cause interference,
capacity limits, or forgetting; these are real problems, but not an unavoidable
consequence of having many references.

PPO is not immune to missing information. If episodes demand different motions
but the policy cannot observe which was selected, conflicting tracking rewards
can produce a compromise or failed learning. More environments do not resolve
that ambiguity.

Diffusion also commonly uses an MSE denoising objective. Its advantage is the
conditional, iterative sampling construction, rather than simply avoiding
squared-error losses.

### Implication for our current observation design

Our current VAE encoder receives reference joint positions and velocities plus
anchor orientation error: 68 dimensions, without reference preview. The selected
teachers also have reference lookahead disabled.

Two motions could share the current features yet require different preparation
for what follows. If this becomes a failure case, possible remedies include a
short reference preview or a persistent intent/phase representation.

Adding preview only to a future teacher would not resolve the student's ambiguity.
The student must also receive enough information to reproduce the teacher's
choices. A generative policy can sample plausible alternatives, but cannot recover
an externally specified intention that was never communicated. Such changes would
require an explicit future schema decision, not a silent change to M2.

## 4. A practical route to a general tracker

Multi-motion PPO is a reasonable baseline. The emphasis should be on conditioning,
data quality, coverage, and capacity rather than maximum parallel environment
count.

1. **Use one selected reference per environment.** Sample a motion and a starting
   phase, maintain an independent clock, and compute rewards against that
   environment's reference. Do not average poses across clips.
2. **Communicate intent sufficiently.** Start with reference-conditioned inputs;
   use preview if needed for anticipation. Clip IDs can disambiguate training
   examples but are not a substitute for a reference representation that
   generalizes to unseen motions.
3. **Balance clips and phases.** Avoid domination by long, easy clips. Combine
   broad coverage with bounded failure-based sampling. Pure hard-example sampling
   can waste training on infeasible retargeted segments.
4. **Curate feasible references.** Check retargeting, contacts, timing, joint
   limits, and dynamic demands. Compute alone cannot make impossible references
   trackable.
5. **Grow diversity and capacity deliberately.** Start with a small diverse set,
   expand the curriculum, and monitor interference. Increase capacity or consider
   structured experts when evidence supports it.
6. **Train recovery, not just nominal imitation.** Use perturbed initial states
   and disturbances. A teacher that succeeds only near its nominal trajectory can
   give poor labels on student-visited states.
7. **Evaluate per motion and difficult phase.** Compare with specialists on
   root-relative pose, heading, contact quality, failures, and recovery. Report
   both training-library coverage and held-out-reference performance; they are
   different forms of generality.

Specialist supervision can warm-start a shared tracker, followed by joint PPO
fine-tuning if justified. This need not start from scratch or require millions
of simultaneous environments.

### When a general teacher is worth the effort

A strong shared teacher can reduce checkpoint/training-run maintenance, transfer
skills across related references, and provide a more consistent control mapping.
However, consistency is not useful if difficult motions become consistently worse.
Average tracking reward is insufficient evidence to replace strong specialists.

A practical future comparison is a curated multi-motion subset, evaluated against
the existing specialists under matched control and observation contracts. Expand
only if quality, robustness, and maintenance cost justify it.

**The planned shared VAE is already one route toward a reference-conditioned
multi-motion tracker:** its encoder consumes the reference and its decoder drives
the robot. Once trained, the combined model can track multiple references. This
does not establish arbitrary unseen-motion generalization, but means a separate
general-PPO teacher is not a prerequisite for the current architecture.

## 5. What remains unsolved after obtaining a general tracker?

The two conditional problems are different:

\[
\underbrace{\pi(a_t\mid s_t,R)}_{\text{execute a reference}}
\qquad\text{versus}\qquad
\underbrace{p(R\mid s_t,\text{task})}_{\text{choose a suitable motion}}.
\]

A general tracker solves the first problem. It does not automatically provide the
second. A task such as returning a ball does not specify every joint angle, foot
placement, or stroke timing.

Beyond test-time guidance, a generative/planning layer can provide:

### Behavior generation without a complete reference

Generate alternative realizations of a sparse objective instead of requiring a
complete reference trajectory from another system. A universal tracker still
needs something to supply that realization.

### Temporally coherent choices

Sampling an independent left/right decision at each step is not a usable planner.
A trajectory model can represent sustained commitments over a horizon. Likewise,
the VAE's Gaussian latent regularizer is not a temporal skill planner: independent
latent samples need not form a meaningful or executable sequence.

### Predictions of consequences as well as commands

BeyondMimic models state–latent trajectories, not only latent actions. Predicted
future states provide a space in which to evaluate objectives and constraints.
The model supplies a learned joint trajectory prior, not an exact physics model
or a feasibility guarantee.

### Separation of planning and immediate feedback

Latent generation specifies intent while a decoder uses the latest proprioception
to produce actions. This can reduce the problems of blindly executing a stale raw
action sequence and offers a structured alternative to modeling irregular PD
setpoints directly. These are arguments for a latent feedback-controller
architecture, not advantages unique to diffusion.

## 6. Diffusion is a choice, not a requirement

The arguments above justify a planner or generative behavior model; they do not
prove it must be diffusion. Alternatives include:

- Motion retrieval and motion graphs.
- Model-predictive control with an appropriate dynamics/behavior model.
- Autoregressive skill-token or other latent-sequence models.
- Motion-space generation followed by a strong general tracker.
- Direct task-conditioned RL when the task family is sufficiently specified and
  retraining for new objectives is acceptable.

A strong general tracker may eliminate the need for a specialist bank. It does not
automatically supply a generative planner or the desired latent interface. On the
other hand, a motion-space generator plus tracker is a legitimate alternative to
the VAE/state–latent-diffusion stack. The latter must justify its complexity through
control quality, planning flexibility, and runtime behavior.

Diffusion does not guarantee feasibility, preserve every mode, or invent missing
skills. Disconnected training clips do not establish safe transitions between
arbitrary phases. State-estimation errors, inaccurate predictions, limited data,
and aggressive guidance can still cause failures. BeyondMimic itself reports
limitations in motion transitions, history-induced repetitive behaviors, and
short-horizon planning.

## 7. Decision and implications for this project

**Agreed now:** retain the strong existing specialist teachers for the current
VAE distillation experiment.

**Recommended future ordering, not an execution plan:**

1. Establish the shared reference-conditioned controller using trusted labels.
2. Determine whether teacher maintenance, supervision inconsistency, or student
   representation/optimization is the actual bottleneck.
3. Consider a general teacher when scaling or label quality warrants it, using
   per-motion and recovery comparisons against specialists.
4. Pursue a planner/generative model when the objective is to choose, adapt, or
   compose behaviors from sparse task objectives—not merely because multiple
   reference motions exist.
5. Compare latent diffusion with simpler planning/generation alternatives on the
   actual downstream tasks.

M2 is an accepted but untrained tensor/model/replay core. This discussion does not
change its schema, authorize M3, or launch PPO, DAgger, diffusion training, remote
jobs, or hardware execution.

**Bottom line: a general tracker answers “how do I execute this motion?” Diffusion
is one way to answer “which executable motion should happen next?”**

## References

- Luo et al., **Perpetual Humanoid Control for Real-time Simulated Avatars**, ICCV
  2023. [Official paper page](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Perpetual_Humanoid_Control_for_Real-time_Simulated_Avatars_ICCV_2023_paper.html).
  Evidence for broad reference-conditioned tracking in simulation; its progressive
  capacity design is more than naive multi-motion batching.
- **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided
  Diffusion**. [Paper](https://arxiv.org/abs/2508.08241),
  [project](https://beyondmimic.github.io/). The locally reviewed paper describes
  the conditional VAE, state–latent trajectory modeling, feedback decoder, and
  guidance in main pp. 17–21, and limitations in p. 12. Local source:
  `/home/agiuser/Documents/beyondmimic.pdf`.
