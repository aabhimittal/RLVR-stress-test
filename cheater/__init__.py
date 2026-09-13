"""CHEATER -- stress-test a reward verifier before spending RL compute."""
from .audit import AuditConfig, AuditReport, run_audit
from .exploitability import Estimate, estimate, frontier, normalised_xi
from .probes import run_probes
from .report import to_json, to_markdown
from .search import ProgramGRPO, SearchConfig
from .verifier import SafeVerifier

__version__ = "0.1.0"
__all__ = [
    "AuditConfig", "AuditReport", "run_audit", "Estimate", "estimate", "frontier",
    "normalised_xi", "run_probes", "to_json", "to_markdown", "ProgramGRPO",
    "SearchConfig", "SafeVerifier", "__version__",
]
