# Third-party notices

## H3 Motion Context

The standalone Motion Context implementation in this addon is derived from:

<https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context>

Copyright (C) 2026 NikoDemon80.

The derived implementation and patch helpers are modified versions of the
upstream `nodes.py`, `patch_layout.py`, and `patch_payload.py`. They remain
licensed under the GNU General Public License, version 3. The complete GPLv3
license text is included in `LICENSE`.

The modifications make the implementation private to this addon, provide the
Auto-Chain input contract, and avoid registering the upstream node IDs.
