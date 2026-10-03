#!/usr/bin/env python3
"""Bounded offline real-model D3 seam replay (entry point).

See ``mjlab/tasks/tracking/diffusion/replay_integration.py`` for the
implementation and the module docstring for scope limits.  All model and
artifact paths are explicit CLI arguments; no absolute path is embedded here.

The report written to ``--output`` labels the decoder conditioning as a
synthetic fixture, sets ``closed_loop: false``, and never claims that the
generated latents were executed.
"""

from mjlab.tasks.tracking.diffusion.replay_integration import main

if __name__ == "__main__":
  main()
