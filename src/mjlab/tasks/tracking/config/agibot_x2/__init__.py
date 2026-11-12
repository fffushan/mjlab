from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import (
  MotionTrackingOnPolicyRunner,
  MultiMotionTrackingOnPolicyRunner,
)

from .env_cfgs import (
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
  agibot_x2_flat_tracking_env_cfg,
  agibot_x2_flat_tracking_increated_perturbations_measured_servo_env_cfg,
  agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg,
  agibot_x2_multi_motion_tracking_env_cfg,
  agibot_x2_multi_motion_tracking_measured_servo_env_cfg,
)
from .rl_cfg import (
  agibot_x2_multi_motion_ppo_runner_cfg,
  agibot_x2_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-AgiBot-X2",
  env_cfg=agibot_x2_flat_tracking_env_cfg(),
  play_env_cfg=agibot_x2_flat_tracking_env_cfg(play=True),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation",
  env_cfg=agibot_x2_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=agibot_x2_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The tracking task the distillation and closed-loop teacher pipeline uses as its
# ``base_task``: correlated domain randomization with the perturbation and reset
# ranges halved. See docs/source/x2_tennis_distillation.rst.
register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
    "Reduced-Perturbations"
  ),
  env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True),
  play_env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, play=True
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name="agibot_x2_tracking_correlated_dr_reduced_perturbations"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The single-motion teacher behind the multi-motion task: increased
# perturbations with the vendor reference-policy PD gains. The multi-motion
# configuration is built on this one, so the two share a robot, action scale,
# observation set and domain randomization by construction.
register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
    "Vendor-Gains"
  ),
  env_cfg=agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg(),
  play_env_cfg=agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg(
    play=True
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name="agibot_x2_tracking_increated_perturbations_vendor_gains"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The same single-clip task with the measured servo damping. This is the
# ``base_task`` of the measured-servo distillation cohort: the collection
# environment must carry the damping the teacher ONNX declares, or the adapter
# refuses the cohort. Not a deployment gain; see the config docstring.
register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
    "MeasuredServo"
  ),
  env_cfg=agibot_x2_flat_tracking_increated_perturbations_measured_servo_env_cfg(),
  play_env_cfg=(
    agibot_x2_flat_tracking_increated_perturbations_measured_servo_env_cfg(play=True)
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name="agibot_x2_tracking_increated_perturbations_measured_servo"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The multi-motion teacher is not a single-motion task: it has no reference clip to
# embed in the policy export. Its deploy graph is therefore the raw ``obs -> actions``
# model and the reference travels as a separate file, which is what lets one policy
# replay a clip the teacher never trained on. The runner writes that graph plus the
# metadata that declares the enabled preview offset.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion",
  env_cfg=agibot_x2_multi_motion_tracking_env_cfg(),
  play_env_cfg=agibot_x2_multi_motion_tracking_env_cfg(play=True),
  rl_cfg=agibot_x2_multi_motion_ppo_runner_cfg(),
  runner_cls=MultiMotionTrackingOnPolicyRunner,
)

# The measured-servo variant is the same teacher with per-joint damping taken from the
# two joint-step runs. It is a separate task id and not a flag, because a policy trained
# on it is not interchangeable with one trained on the base: the plant differs, and the
# exported metadata must carry the gains it was trained with. The runner is the same
# multi-motion runner, so the export path and the preview contract are unchanged.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion-MeasuredServo",
  env_cfg=agibot_x2_multi_motion_tracking_measured_servo_env_cfg(),
  play_env_cfg=agibot_x2_multi_motion_tracking_measured_servo_env_cfg(play=True),
  rl_cfg=agibot_x2_multi_motion_ppo_runner_cfg(),
  runner_cls=MultiMotionTrackingOnPolicyRunner,
)
