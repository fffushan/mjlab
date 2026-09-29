import pytest

from mjlab.tasks.tracking.diffusion import ContractError, DiffusionContract


def test_frozen_yaml_contract_identity_and_dimensions() -> None:
  contract = DiffusionContract.from_yaml(
    "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
  )
  assert contract.state_dimension == 135
  assert contract.window_steps == 41
  assert contract.token_dimension == 231
  assert (
    contract.contract_sha256
    == "42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7"
  )


def test_artifact_identity_is_fail_closed() -> None:
  contract = DiffusionContract()
  with pytest.raises(ContractError):
    contract.validate_artifacts({})
  contract.validate_artifacts(
    {
      "vae": {
        "role": "vae",
        "path": contract.vae_checkpoint,
        "sha256": contract.vae_sha256,
      },
      "recovery": {
        "role": "recovery",
        "path": contract.recovery_onnx,
        "sha256": contract.recovery_sha256,
      },
      "x2_xml": {
        "role": "x2_xml",
        "path": contract.state_xml,
        "sha256": contract.state_xml_sha256,
      },
      "physical_fall_source": {
        "role": "physical_fall_source",
        "path": contract.physical_fall_source,
        "sha256": contract.physical_fall_source_sha256,
      },
      "vae_tracking_rejection_source": {
        "role": "vae_tracking_rejection_source",
        "path": contract.vae_tracking_rejection_source,
        "sha256": contract.vae_tracking_rejection_source_sha256,
      },
    }
  )


def test_source_identity_mismatch_and_missing_fail_closed() -> None:
  contract = DiffusionContract()
  artifacts = {
    role: {"role": role, "path": path, "sha256": digest}
    for role, path, digest in (
      ("vae", contract.vae_checkpoint, contract.vae_sha256),
      ("recovery", contract.recovery_onnx, contract.recovery_sha256),
      ("x2_xml", contract.state_xml, contract.state_xml_sha256),
      (
        "physical_fall_source",
        contract.physical_fall_source,
        contract.physical_fall_source_sha256,
      ),
      (
        "vae_tracking_rejection_source",
        contract.vae_tracking_rejection_source,
        contract.vae_tracking_rejection_source_sha256,
      ),
    )
  }
  missing = dict(artifacts)
  del missing["x2_xml"]
  with pytest.raises(ContractError):
    contract.validate_artifacts(missing)
  mismatch = dict(artifacts)
  mismatch["x2_xml"] = {
    **mismatch["x2_xml"],
    "sha256": "0" * 64,
  }
  with pytest.raises(ContractError):
    contract.validate_artifacts(mismatch)
