"""Reusable research experiment report generator (Phase 9).

Standardizes evaluation reporting across computational and human validation studies.

Core Principle:
Every report must explicitly separate:
1. Observed Result (factual metric from run)
2. Interpretation (reasoned explanation)
3. Scientific Claim (strictly what is empirically justified)
4. Limitations (clear boundaries of validity)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ResearchReport:
    """Standardized research evaluation report structure."""

    experiment_id: str
    title: str
    research_question: str
    hypothesis: str
    dataset_identifier: str
    sample_size: int
    procedure: str
    metrics: dict[str, Any]
    observed_results: dict[str, Any]
    interpretation: str
    scientific_claims: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render report as GitHub-flavored Markdown document."""
        lines = [
            f"# {self.experiment_id}: {self.title}",
            "",
            f"**Timestamp:** `{self.timestamp}`  ",
            f"**Dataset:** `{self.dataset_identifier}` ($N = {self.sample_size}$)  ",
            "",
            "## 1. Research Question",
            f"> {self.research_question}",
            "",
            "## 2. Hypothesis",
            f"{self.hypothesis}",
            "",
            "## 3. Experimental Procedure",
            f"{self.procedure}",
            "",
            "## 4. Observed Results",
            "```json",
            json.dumps(self.observed_results, indent=2),
            "```",
            "",
            "## 5. Scientific Interpretation",
            f"{self.interpretation}",
            "",
            "## 6. Supported Claims",
        ]
        if self.scientific_claims:
            for claim in self.scientific_claims:
                lines.append(f"- [x] {claim}")
        else:
            lines.append("- *No causal claims supported at this stage.*")

        lines.extend([
            "",
            "## 7. Limitations & Threats to Validity",
        ])
        if self.limitations:
            for lim in self.limitations:
                lines.append(f"- [!] {lim}")
        else:
            lines.append("- *No limitations declared.*")

        lines.extend([
            "",
            "## 8. Reproducibility Metadata",
            "```json",
            json.dumps(self.reproducibility, indent=2),
            "```",
            "",
        ])

        return "\n".join(lines)

    def save(self, output_path: str | Path) -> None:
        """Save report to disk in JSON and/or Markdown format depending on extension."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".md":
            p.write_text(self.to_markdown(), encoding="utf-8")
        else:
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
