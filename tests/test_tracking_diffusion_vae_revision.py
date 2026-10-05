"""Diffusion contract gates: VAE repins, fail-closed registries.

Revisions 2 and 3 keep every representation, window, split and normalization
field of revision 1 and repin only the frozen VAE (the 12-teacher and then the
30-teacher increased-perturbation students).  These tests prove each widening is
deliberate and still fail-closed: unregistered contract files and unapproved VAE
identities must be refused, every earlier revision stays loadable, and the
rollout selection may only narrow the checkpoint's own cohort.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from mjlab.tasks.tracking.diffusion import ContractError, DiffusionContract
from mjlab.tasks.tracking.diffusion.runtime_x2 import (
  RuntimeX2Error,
  resolve_rollout_ids,
)

D0 = Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")
D1 = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")
D0_INC12 = Path("docs/plans/beyondmimic_diffusion_d0_inc12_contract.yaml")
D1_INC12 = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_inc12_contract.yaml")
D0_INC30 = Path("docs/plans/beyondmimic_diffusion_d0_inc30_contract.yaml")
D1_INC30 = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_inc30_contract.yaml")

INC12_VAE = "logs/distillation/vae-inc-a-12t-nc/checkpoint-final.pt"
INC12_VAE_SHA = "22a650bcba7375bca4578c229554ea3c93c357244919ea83d7ba1b6878282a4d"
INC12_COHORT_MANIFEST = (
  "configs/distillation/x2_tennis_increased_perturbation_a_v1.yaml"
)
INC12_COHORT_IDS = (
  "inc000",
  "inc001",
  "inc002",
  "inc003",
  "inc004",
  "inc005",
  "inc006",
  "inc007",
  "inc008",
  "inc009",
  "inc010",
  "inc011",
)
# Registered contract-file digests (contract.py `_CONTRACT_SHAS_BY_SCHEMA`).
FROZEN_DIGESTS = {
  D0: "42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7",
  D1: "f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed",
  D0_INC12: "39f1e85370efebdf83716d3e60ee505c1b38057c88bce3527439ad949ad23d9b",
  D1_INC12: "645fef76f672408850832ab9bb06e0436cca517473b4a1d74004182138954d05",
  D0_INC30: "14319f177b9022205365c472544dd57850c00f04cf16214341f0c013cfc1911a",
  D1_INC30: "95a72864773087b8ee711da14d324ca2c26b6654fea92122026987038d126202",
}

INC30_VAE = "logs/distillation/vae-inc-all-30t-nc/checkpoint-final.pt"
INC30_VAE_SHA = "4a90bbdf76028d3cbe56cfcfbce9604ef96ce7b5ab5e0a7c241235be6f2dee0e"
INC30_COHORT_MANIFEST = (
  "configs/distillation/x2_tennis_increased_perturbation_all_v1.yaml"
)
INC30_COHORT_IDS = (
  "inc000",
  "inc001",
  "inc002",
  "inc003",
  "inc004",
  "inc005",
  "inc006",
  "inc007",
  "inc008",
  "inc009",
  "inc010",
  "inc011",
  "inc013",
  "inc014",
  "inc015",
  "inc016",
  "inc017",
  "inc018",
  "inc019",
  "inc020",
  "inc021",
  "inc022",
  "inc023",
  "inc024",
  "inc025",
  "inc026",
  "inc027",
  "inc028",
  "inc029",
  "inc030",
)


@pytest.mark.parametrize("path", [D0, D1, D0_INC12, D1_INC12, D0_INC30, D1_INC30])
def test_registered_revisions_load_with_their_pinned_digest(path: Path) -> None:
  contract = DiffusionContract.from_yaml(path)
  assert contract.contract_sha256 == FROZEN_DIGESTS[path]


def test_inc12_revision_repins_only_the_vae() -> None:
  revision_1 = DiffusionContract.from_yaml(D1)
  revision_2 = DiffusionContract.from_yaml(D1_INC12)
  assert revision_2.vae_checkpoint == INC12_VAE
  assert revision_2.vae_sha256 == INC12_VAE_SHA
  assert revision_2.schema_version == revision_1.schema_version
  assert revision_2.token_dimension == revision_1.token_dimension == 167
  assert revision_2.projection_rows == revision_1.projection_rows == 0
  ignored = {"contract_sha256", "vae_checkpoint", "vae_sha256"}
  differing = {
    name
    for name in revision_1.__slots__
    if name not in ignored and getattr(revision_1, name) != getattr(revision_2, name)
  }
  assert differing == set(), (
    f"revision 2 changed fields it must not: {sorted(differing)}"
  )


def test_inc12_declares_the_twelve_member_cohort_without_real_robot_claim() -> None:
  payload = yaml.safe_load(D1_INC12.read_text())
  vae = payload["vae"]
  assert vae["cohort_manifest"] == INC12_COHORT_MANIFEST
  assert tuple(vae["cohort_ids"]) == INC12_COHORT_IDS
  assert vae["dimensions"] == {
    "encoder_command": 68,
    "conditioning": 99,
    "latent": 32,
    "action": 31,
  }
  # The 12-teacher student has no real-robot validation; the contract says so.
  assert vae["user_confirmed_real_robot_validation"] is False


def test_unregistered_contract_revision_fails_closed(tmp_path: Path) -> None:
  # Byte-identical content is registered; a single added comment is not.
  altered = tmp_path / "beyondmimic_diffusion_d1_no_mixtures_inc12_contract.yaml"
  altered.write_text(D1_INC12.read_text() + "\n")
  assert hashlib.sha256(altered.read_bytes()).hexdigest() != str(
    FROZEN_DIGESTS[D1_INC12]
  )
  with pytest.raises(ContractError, match="contract hash mismatch"):
    DiffusionContract.from_yaml(altered)


def test_unapproved_vae_identity_fails_closed() -> None:
  payload = yaml.safe_load(D0_INC12.read_text())
  payload["vae"]["sha256"] = "0" * 64
  with pytest.raises(ContractError, match="approved revision"):
    DiffusionContract.from_mapping(payload)


def test_approved_vae_identity_is_accepted_but_path_must_match() -> None:
  payload = yaml.safe_load(D0_INC12.read_text())
  # Same digest, different declared path: still refused, because the approved
  # pair is (path, sha256).
  payload["vae"]["checkpoint"] = "logs/distillation/somewhere-else/checkpoint-final.pt"
  with pytest.raises(ContractError, match="approved revision"):
    DiffusionContract.from_mapping(payload)
  # Restoring the registered path loads.
  payload["vae"]["checkpoint"] = INC12_VAE
  contract = DiffusionContract.from_mapping(payload)
  assert contract.vae_sha256 == INC12_VAE_SHA


def test_inc30_revision_repins_only_the_vae() -> None:
  # Revision 3 moves the VAE identity and nothing else, exactly as revision 2 did.
  revision_2 = DiffusionContract.from_yaml(D1_INC12)
  revision_3 = DiffusionContract.from_yaml(D1_INC30)
  assert revision_3.vae_checkpoint == INC30_VAE
  assert revision_3.vae_sha256 == INC30_VAE_SHA
  assert revision_3.token_dimension == revision_2.token_dimension == 167
  ignored = {"contract_sha256", "vae_checkpoint", "vae_sha256"}
  differing = {
    name
    for name in revision_2.__slots__
    if name not in ignored and getattr(revision_2, name) != getattr(revision_3, name)
  }
  assert differing == set(), (
    f"revision 3 changed fields it must not: {sorted(differing)}"
  )


def test_inc30_declares_the_thirty_member_cohort_without_real_robot_claim() -> None:
  payload = yaml.safe_load(D1_INC30.read_text())
  vae = payload["vae"]
  assert vae["cohort_manifest"] == INC30_COHORT_MANIFEST
  assert tuple(vae["cohort_ids"]) == INC30_COHORT_IDS
  assert len(INC30_COHORT_IDS) == 30
  # Motion 012 has no teacher run anywhere: the cohort skips it explicitly.
  assert "inc012" not in vae["cohort_ids"]
  assert vae["dimensions"] == {
    "encoder_command": 68,
    "conditioning": 99,
    "latent": 32,
    "action": 31,
  }
  assert vae["user_confirmed_real_robot_validation"] is False


def test_inc30_pinned_manifest_and_provenance_digests_match_the_files() -> None:
  """The pin must match the bytes on disk, not a remembered value.

  `logs/` is Git-ignored, so a fresh clone or a clean worktree legitimately has
  no artifact tree.  That is a missing input, not a regression: skip with the
  source named rather than failing.  Where the artifacts are present (the NAS
  tree and any full mirror) the check is a real byte comparison.
  """
  payload = yaml.safe_load(D1_INC30.read_text())
  vae = payload["vae"]
  manifest = Path(vae["cohort_manifest"])
  provenance = Path(vae["source_hash_index"])
  missing = [str(path) for path in (manifest, provenance) if not path.exists()]
  if missing:
    pytest.skip(
      "ignored artifact tree absent (present on the NAS tree): " + ", ".join(missing)
    )
  assert hashlib.sha256(manifest.read_bytes()).hexdigest() == str(
    vae["cohort_manifest_sha256"]
  )
  assert hashlib.sha256(provenance.read_bytes()).hexdigest() == str(
    vae["source_hash_index_sha256"]
  )


def test_inc30_unregistered_contract_revision_fails_closed(tmp_path: Path) -> None:
  altered = tmp_path / "beyondmimic_diffusion_d1_no_mixtures_inc30_contract.yaml"
  altered.write_text(D1_INC30.read_text() + "\n")
  with pytest.raises(ContractError, match="contract hash mismatch"):
    DiffusionContract.from_yaml(altered)


def test_rollout_selection_defaults_to_every_member() -> None:
  assert resolve_rollout_ids(INC12_COHORT_IDS, "") == INC12_COHORT_IDS
  assert resolve_rollout_ids(INC12_COHORT_IDS, "   ") == INC12_COHORT_IDS


def test_rollout_selection_may_narrow_to_a_subset_in_order() -> None:
  assert resolve_rollout_ids(INC12_COHORT_IDS, "inc002,inc000") == ("inc002", "inc000")
  assert resolve_rollout_ids(("a", "b"), " a , b ") == ("a", "b")


@pytest.mark.parametrize(
  "selection,match",
  [
    ("inc000,inc000", "repeats a member"),
    ("inc012", "not members"),
    ("inc000,tennis_001", "not members"),
  ],
)
def test_rollout_selection_rejects_unknown_or_repeated_members(
  selection: str, match: str
) -> None:
  with pytest.raises(RuntimeX2Error, match=match):
    resolve_rollout_ids(INC12_COHORT_IDS, selection)


def test_rollout_selection_rejects_an_empty_cohort() -> None:
  with pytest.raises(RuntimeX2Error, match="no members"):
    resolve_rollout_ids((), "")


def test_three_clip_rollout_on_the_twelve_teacher_manifest_is_a_subset() -> None:
  # The matched three-clip arm rolls out inc000/inc001/inc002 out of twelve.
  assert resolve_rollout_ids(INC12_COHORT_IDS, "inc000,inc001,inc002") == (
    "inc000",
    "inc001",
    "inc002",
  )
