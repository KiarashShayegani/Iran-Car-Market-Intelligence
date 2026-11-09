"""Collector registry: maps config source keys to collector classes.

Adding a platform means adding one collector module and registering it here.
No source logic lives outside collectors/.
"""

from __future__ import annotations

from icmi.collectors.bama import BamaCollector
from icmi.collectors.base import BaseCollector
from icmi.collectors.hamrah_mechanic import HamrahMechanicCollector
from icmi.collectors.sheypoor import SheypoorCollector

COLLECTORS: dict[str, type[BaseCollector]] = {
    BamaCollector.source_key: BamaCollector,
    SheypoorCollector.source_key: SheypoorCollector,
    HamrahMechanicCollector.source_key: HamrahMechanicCollector,
}


def get_collector(source_key: str) -> type[BaseCollector]:
    try:
        return COLLECTORS[source_key]
    except KeyError:
        available = ", ".join(sorted(COLLECTORS)) or "(none registered)"
        raise KeyError(
            f"No collector registered for '{source_key}'. Available: {available}"
        ) from None
