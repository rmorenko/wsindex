"""WSIndex — a learning CLI that semantically indexes a developer workspace.

The entry point lives in `wsindex.cli`. What the tool does and what it
costs is in the README; the decisions behind it are in `docs/adr/`, with
the early ones and the quality attributes in ARCH_en.md.

Logging follows the rule libraries are supposed to follow and this one
did not: modules log, the package handles nothing, and whoever embeds it
decides where that goes. The NullHandler below is what makes "log
freely" safe — without it Python prints its own complaint the first time
a record has nowhere to go.

Two callers, two answers. `wsindex search` is run by a person, so the
CLI keeps writing sentences to stderr and turns none of this on: a log
line is not an interface for somebody watching a terminal. `wsindex
serve` runs unattended and its reader arrives afterwards, so it does
turn it on — a server that kept nothing could not answer "was it broken
last night".
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
