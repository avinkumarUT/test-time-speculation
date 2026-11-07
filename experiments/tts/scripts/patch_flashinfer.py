"""Enlarge FlashInfer's prefill plan workspace from 8 MB to 64 MB.

FlashInfer's batch-prefill wrappers allocate a fixed 8 MB integer workspace
for the attention plan. Verifying a 16-token DFlash block for 64 concurrent
requests needs a larger plan than that and fails with a buffer-overflow
error. The size is hard-coded in the installed package, so this script edits
the installed `flashinfer/prefill.py` in the active Python environment. It
only affects capacity, not numerics, and is needed only for the bs=64 DFlash
cells.

    python experiments/tts/scripts/patch_flashinfer.py            # apply
    python experiments/tts/scripts/patch_flashinfer.py --revert   # undo

Idempotent. Tested with flashinfer 0.6.11.post1 (the version pinned by this
SGLang release); the script refuses to edit a file it does not recognize.
"""

import argparse
import importlib.metadata
import importlib.util
import os

TEMPLATE = (
    "        self._int_workspace_buffer = torch.empty(\n"
    "            ({mb} * 1024 * 1024,), dtype=torch.uint8, device=self.device\n"
    "        )\n"
)
STOCK, PATCHED = TEMPLATE.format(mb=8), TEMPLATE.format(mb=64)
EXPECTED_SITES = 2  # BatchPrefillWithPagedKVCacheWrapper, ...WithRaggedKVCacheWrapper


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--revert", action="store_true", help="Restore the 8 MB size.")
    args = ap.parse_args()

    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("flashinfer is not installed in this environment")
    path = os.path.join(list(spec.submodule_search_locations)[0], "prefill.py")
    version = importlib.metadata.version("flashinfer-python")
    with open(path) as f:
        src = f.read()

    old, new = (PATCHED, STOCK) if args.revert else (STOCK, PATCHED)
    if src.count(new) == EXPECTED_SITES and src.count(old) == 0:
        print(f"nothing to do: {path} (flashinfer {version}) is already "
              f"{'stock' if args.revert else 'patched'}")
        return
    if src.count(old) != EXPECTED_SITES:
        raise SystemExit(
            f"cannot edit {path} (flashinfer {version}): expected "
            f"{EXPECTED_SITES} workspace allocations of the known form, found "
            f"{src.count(old)}. This flashinfer version needs a manual look.")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"{'reverted' if args.revert else 'patched'} {path} (flashinfer {version})")


if __name__ == "__main__":
    main()
