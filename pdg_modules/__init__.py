"""
PDG MCP Server Modules

Modular organization mirroring the PDG API structure:
- api: Core API functionality (search, properties, etc.)
- data: Data handling and measurements
- decay: Decay analysis and branching fractions  
- errors: Error handling and diagnostics
- measurement: PDG measurement objects and analysis
- particle: PDG particle objects and quantum numbers
- units: Unit conversions and physics constants
- utils: PDG utility functions and data processing
"""

from . import api
from . import data
from . import decay
from . import errors
from . import measurement
from . import particle
from . import units
from . import utils

__version__ = "1.0.0"
__all__ = ["api", "data", "decay", "errors", "measurement", "particle", "units", "utils"] 