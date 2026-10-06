# D5a CPU latent runtime — accepted slice

Observation and parent acceptance: 2026-10-06.

## Status

The parent accepts this slice for CPU software implementation and validation.
The final independent review returned **OK with notes** and found no current
source or test-coverage blocker.

This is not full D5 acceptance.
Sustained 50 Hz timing, trained-model GPU execution, export, physics, controller,
vendor sim2sim, deployment and hardware readiness are not tested here.
No physical stability or model-quality improvement is claimed.

Worktree used for validation:
`/home/agiuser/projects/worktrees/mjlab/d5-runtime-20261006/integration`.
Base commit: `3d6a8e5b72a81de9f3332e83ac1b3c469d51c98d`.
The base is provenance. It was not an execution gate.

The accepted source, tests, README, changelog and CPU evidence are now copied
into the primary repository as the scoped commit requested after acceptance.
The original isolated worktree and its source archive remain preserved.
No push has occurred.

## Implementation

- `scheduler.py`: latest pending request, one active request, immutable context,
  accepted-request provenance, elapsed-tick fallback and lifecycle invalidation.
- `latent_planning.py`: owned 41 x 167 planning packets, D3/D4 sampling reuse,
  request-local RNG and raw 33 x 32 latent trajectories.
- `planner_executor.py`: one owned spawn worker, bounded request/result transport,
  strict wire attribution and exact bounded shutdown.
- `async_latent_runtime.py`: execution-tick row selection, source-age and identity
  checks, current 99-D decoder conditioning, finite 31-D actions and
  acknowledgement-gated measured history.

The generic selector carries raw latent vectors internally in `Plan.actions`.
Explicit metadata and the runtime adapter prevent treating those vectors as
robot commands. The runtime does not predecode future actions.

The planner has no post-bootstrap encoder or decoder call.
The runtime decodes only the selected row with current proprioception.
D4 keyframes and pose guidance use the existing component mathematics.

## Final validation

Parent validation used CUDA-disabled synthetic CPU inputs and callbacks.

| Gate | Result |
| --- | --- |
| Optimized executor tests | 39 passed in 45.89 s |
| Remaining seven D5/adjacent test files | 163 passed in 15.63 s |
| Unique focused test coverage | 202 passed |
| Affected scheduler file after type-only correction | 71 passed in 1.33 s |
| Ruff format and lint, all eight source/test files | Passed |
| Ty, all eight source/test files | Passed |
| Pyright, all eight source/test files | 0 errors, 0 warnings |
| Untracked-aware whitespace and tracked diff checks | Passed |
| No staged files | Verified |
| Primary revalidation: optimized executor group | 39 passed in 50.62 s |
| Primary revalidation: remaining D5/adjacent group | 163 passed in 15.81 s |
| Primary scoped Ruff format and lint | Passed |
| Primary scoped Ty and Pyright | 0 errors, 0 warnings |

The two test groups had unchanged 120-second bounds.
The real spawned CPU `LatentPlanner.process` test is included.
It tests actual sampler/denormalization, current-state decoding,
acknowledgement history and worker cleanup.

The final test changes are covered by the final independent review.
The scheduler type correction changed only how its ownership test accesses the
same list and array references. The affected file was rerun afterward.
The other seven source/test hashes match the passing split validation.
The full repository test suite was not run.

## Evidence and source

- [Final independent review](evidence/independent-review.md)
- [Executor test log](evidence/executor-tests.log)
- [Runtime/tensor test log](evidence/runtime-tensor-tests.log)
- [Affected-test and static-check log](evidence/type-fix-and-static-checks.log)
- [Intermediate static failure](evidence/intermediate-static-failure.log)
- [Final ten-file manifest](source/final-source.sha256)
- [Final source archive](source/final-source.tar.gz)
- [Tracked changelog patch](source/changelog.patch)

Archive SHA-256:
`f8bdebff8f2042824e2cdcef1c22f8d15b6d84c2effc6aea23b672a88a5f99ec`.

Final reviewer run: `4f65f70d-93f8-4d31-adf9-4797a313e6fd`.
All delegated implementation/review roles used `lz-fast/gpt-5.6-luna:max`.
Mission: `9062d164-5167-434a-9ffd-d0d3537b16eb`.

## Historical failures remain separate

1. The initial workflow failed on an undefined status-serializer field.
   Completed component source and handoffs were preserved.
2. The initial review blocked request-origin binding, wire attribution, cleanup
   evidence, execution IDs and real-planner process coverage. Those fixes passed
   final source review.
3. A native verification timed out at 121.014 s against 120 s.
   It remains failed evidence. The full command was not retried with more time.
4. The integration worker retained an older 37-test file instead of the optimized
   39-test file. The parent copied the exact optimized file and checked its hash.
5. Parent monitor #3 passed both CPU groups but then failed broader Ty checks.
   Monitor #3 remains error/exit 1. The corrected affected-test and static-check
   run is separate passing evidence.

The detailed history, manifests, archives and logs remain under
`../../../../worktrees/mjlab/d5-runtime-20261006/orchestration/` in this D5
runtime workstream.

## Remaining boundaries

- Primary integration is complete for this accepted CPU slice. The scoped source,
  tests, README, changelog, plan and evidence are included in the local commit.
  The original isolated worktree and source archive remain preserved.
- Actual frozen model loading and realistic backend execution are not qualified
  by synthetic tests.
- Declare the GPU worker/decoder topology before measuring timing and memory.
- Measure the complete path and source age. A 50 Hz decoder with a slower planner
  is multi-rate, not 50 Hz replanning.
- Export parity and the pose-guidance derivative backend remain separate work.
- Simulation timing, disturbances, delay/dropout, release and vendor sim2sim
  need separately scoped validation.
- Hardware operation remains D6 and needs separate authorization.

Next action: scope actual-model backend and timing validation. This report does
not authorize GPU timing, publication, physics, controller, export, vendor
sim2sim, deployment or hardware execution.
