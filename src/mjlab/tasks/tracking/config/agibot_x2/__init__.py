from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
  agibot_x2_flat_tracking_env_cfg,
  agibot_x2_flat_tracking_observation_ablation_env_cfg,
)
from .rl_cfg import agibot_x2_tracking_ppo_runner_cfg

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

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR",
  env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(),
  play_env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(play=True),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name="agibot_x2_tracking_correlated_dr"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

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
register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
    "Reduced-Perturbations-Projected-Gravity"
  ),
  env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg("projected_gravity"),
  play_env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "projected_gravity", play=True
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity"
    )
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
    "Reduced-Perturbations-Projected-Gravity-And-Anchor"
  ),
  env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "projected_gravity_anchor"
  ),
  play_env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "projected_gravity_anchor", play=True
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity_anchor"
    )
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
    "Reduced-Perturbations-Vendor-Velocity-Scaling"
  ),
  env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "vendor_velocity_scaling"
  ),
  play_env_cfg=agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "vendor_velocity_scaling", play=True
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_tracking_correlated_dr_reduced_perturbations_vendor_velocity_scaling"
    )
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
    "Reduced-Perturbations-Pelvis-Anchor"
  ),
  env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, anchor_body_name="pelvis"
  ),
  play_env_cfg=agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, play=True, anchor_body_name="pelvis"
  ),
  rl_cfg=agibot_x2_tracking_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_tracking_correlated_dr_reduced_perturbations_pelvis_anchor"
    )
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
