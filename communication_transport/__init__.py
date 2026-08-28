"""Operator-valued communication and transport experiments for small transformers.

The package deliberately keeps the carrier-agnostic transformer analysis separate
from the optional, marked Albert control branch.  See ``communication_transport.run``
for the integrated command-line entry point.
"""

from .config import RunConfig

__all__ = ["RunConfig"]
