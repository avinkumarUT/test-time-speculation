"""Per-request acceptance log, enabled by --speculative-accept-log-dir.

Each request gets a file <dir>/<rid>.txt with one line per verify round: that
round's num_correct_drafts, bonus token excluded. This gives acceptance length
as a function of output length; examples/tts/accept_length_curve.py tabulates it.

The log is written at verify time. Under the overlap schedule of EAGLE-3 the next
round has already run when a request's last result is processed, so each file
ends with one extra round whose tokens are discarded.
"""

import logging
import os
import re
from typing import Iterable, Optional

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

logger = logging.getLogger(__name__)

# The request id is supplied by the client; every other character, including
# path separators, is replaced.
_UNSAFE_RID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


class AcceptLog:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._enabled = True

    def write(self, reqs: Iterable, num_correct_drafts: Iterable[int]) -> None:
        if not self._enabled:
            return
        try:
            for req, num_correct in zip(reqs, num_correct_drafts):
                if str(req.rid).startswith(HEALTH_CHECK_RID_PREFIX):
                    continue
                rid = _UNSAFE_RID_CHARS.sub("_", str(req.rid))[:200]
                with open(os.path.join(self.log_dir, f"{rid}.txt"), "a") as f:
                    f.write(f"{int(num_correct)}\n")
        except OSError as e:
            # A full disk must not take the server down.
            logger.warning("Acceptance log disabled: %s", e)
            self._enabled = False


def make_accept_log(server_args, tp_rank: int) -> Optional[AcceptLog]:
    """The log for this rank; only TP rank 0 writes, since all TP ranks agree."""
    log_dir = getattr(server_args, "speculative_accept_log_dir", None)
    return AcceptLog(log_dir) if log_dir and tp_rank == 0 else None
