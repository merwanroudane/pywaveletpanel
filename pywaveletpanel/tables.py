"""
Journal-quality table formatting for econometric results.

Produces tables matching the style of top economics journals
(Econometrica, AER, Energy Journal, SNDE) with:
- Significance stars (*** 1%, ** 5%, * 10%)
- Standard errors in parentheses or t-statistics
- Multi-column regression tables
- LaTeX, HTML, and console (rich) output

Author: Dr. Merwan Roudane (merwanroudane920@gmail.com)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO

try:
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich.text import Text
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from tabulate import tabulate


def _stars(pval: float) -> str:
    """Significance stars."""
    if pval <= 0.01:
        return "***"
    elif pval <= 0.05:
        return "**"
    elif pval <= 0.10:
        return "*"
    return ""


def _fmt(val: float, dec: int = 3) -> str:
    """Format a number."""
    return f"{val:.{dec}f}"


# ══════════════════════════════════════════════════════════════════════════════
# Regression Table (Papers 2, 4 style)
# ══════════════════════════════════════════════════════════════════════════════

class RegressionTable:
    """
    Multi-column regression table in journal style.

    Modeled after Table 1 in Gallegati et al. (2015) and
    Tables 2–3 in Karlsson et al. (2020).

    Layout:
    ┌─────────────┬───────────┬──────────┬──────────┬──────────┐
    │             │ Aggregate │ S3       │ D3       │ D2       │ ...
    │ β           │ -0.599*** │ -0.968***│  0.360*  │  0.191** │ ...
    │             │ (-4.22)   │ (-4.16)  │ (1.70)   │ (2.48)   │ ...
    │ Adj. R²     │ 0.4170    │ 0.6188   │ 0.0399   │ 0.0809   │ ...
    └─────────────┴───────────┴──────────┴──────────┴──────────┘
    """

    def __init__(self, df: pd.DataFrame, title: str = ""):
        self.df = df
        self.title = title

    @classmethod
    def from_scale_result(cls, result, decimals: int = 3) -> "RegressionTable":
        """Build from a ScaleRegressionResult."""
        rows = []
        columns = []

        # Column order: Aggregate, S_J, D_J, ..., D_1
        if result.aggregate_result is not None:
            columns.append(("Aggregate", "Raw data", result.aggregate_result))
        columns.append((
            f"S{result.level}",
            result.scale_labels.get(f"S{result.level}", ""),
            result.scale_results.get(f"S{result.level}"),
        ))
        for j in range(result.level, 0, -1):
            key = f"D{j}"
            if key in result.scale_results:
                columns.append((
                    key,
                    result.scale_labels.get(key, ""),
                    result.scale_results[key],
                ))

        # Build DataFrame
        col_names = [c[0] for c in columns]
        freq_labels = [c[1] for c in columns]

        data = {}
        for p in range(len(result.regressor_names)):
            name = result.regressor_names[p]
            coef_row = {}
            tstat_row = {}
            for col_name, _, res in columns:
                if res is not None:
                    c = res["coef"][p]
                    t = res["t_stat"][p]
                    pv = res["pvalue"][p]
                    coef_row[col_name] = f"{c:.{decimals}f}{_stars(pv)}"
                    tstat_row[col_name] = f"({t:.2f})"
                else:
                    coef_row[col_name] = "—"
                    tstat_row[col_name] = ""
            rows.append({"Row": f"β ({name})", **coef_row})
            rows.append({"Row": "", **tstat_row})

        # R² row
        r2_row = {"Row": "Adj. R²"}
        for col_name, _, res in columns:
            if res is not None:
                r2_row[col_name] = f"{res.get('adj_r_squared', res['r_squared']):.4f}"
            else:
                r2_row[col_name] = "—"
        rows.append(r2_row)

        # N obs row
        nobs_row = {"Row": "N obs"}
        for col_name, _, res in columns:
            if res is not None:
                nobs_row[col_name] = str(res["nobs"])
            else:
                nobs_row[col_name] = "—"
        rows.append(nobs_row)

        df = pd.DataFrame(rows).set_index("Row")
        title = (
            "Scale-by-scale Panel Regression Results\n"
            f"Wavelet: {result.wavelet.upper()} | Level: {result.level} | "
            f"N={result.n_entities}, T={result.n_periods}\n"
            "Fixed effects estimation with Newey-West robust standard errors.\n"
            "t-statistics in parentheses. *** p<0.01, ** p<0.05, * p<0.10"
        )
        return cls(df, title=title)

    def render(self) -> str:
        """Render as a formatted string table."""
        if HAS_RICH:
            return self._render_rich()
        return self._render_tabulate()

    def _render_rich(self) -> str:
        """Render using rich library for beautiful console output."""
        console = Console(record=True, width=120)
        table = RichTable(
            title=self.title,
            title_style="bold cyan",
            border_style="bright_blue",
            header_style="bold magenta",
            show_lines=True,
            pad_edge=True,
        )
        table.add_column("", style="bold", min_width=12)
        for col in self.df.columns:
            table.add_column(col, justify="center", min_width=12)

        for idx, row in self.df.iterrows():
            values = [str(idx)] + [str(v) for v in row.values]
            # Highlight significant coefficients
            styled = []
            for v in values:
                if "***" in str(v):
                    styled.append(f"[bold green]{v}[/]")
                elif "**" in str(v):
                    styled.append(f"[green]{v}[/]")
                elif "*" in str(v) and "(" not in str(v):
                    styled.append(f"[yellow]{v}[/]")
                else:
                    styled.append(str(v))
            table.add_row(*styled)

        console.print(table)
        return console.export_text()

    def _render_tabulate(self) -> str:
        """Render using tabulate as fallback."""
        header = f"\n{'═' * 80}\n{self.title}\n{'═' * 80}\n"
        body = tabulate(
            self.df.reset_index(),
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False,
        )
        footer = f"\n{'═' * 80}\n"
        return header + body + footer

    def to_latex(self) -> str:
        """Export as a LaTeX table."""
        lines = []
        ncols = len(self.df.columns)
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + self.title.split("\\n")[0] + "}")
        lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
        lines.append(r"\hline\hline")
        # Header
        header = " & ".join([""] + list(self.df.columns)) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        # Body
        for idx, row in self.df.iterrows():
            vals = [str(idx)] + [str(v) for v in row.values]
            lines.append(" & ".join(vals) + r" \\")
        lines.append(r"\hline\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\parbox{\textwidth}{\footnotesize Note: *** p$<$0.01, ** p$<$0.05, * p$<$0.10. "
                      r"t-statistics in parentheses.}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def to_html(self) -> str:
        """Export as styled HTML table."""
        return self.df.to_html(classes="table table-striped table-bordered")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.df.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Unit Root Comparison Table (Papers 3, 5 style)
# ══════════════════════════════════════════════════════════════════════════════

class UnitRootTable:
    """
    Comparison table for panel unit root tests.

    Modeled after Table 7 in Almasri et al. (2016).
    """

    def __init__(self, df: pd.DataFrame, title: str = ""):
        self.df = df
        self.title = title

    @classmethod
    def from_single_result(cls, result) -> "UnitRootTable":
        """Build from a single UnitRootResult."""
        rows = [
            {"": "Test statistic", result.test_name: f"{result.statistic:.4f}"},
            {"": "p-value", result.test_name: f"{result.pvalue:.4f}{_stars(result.pvalue)}"},
        ]
        for alpha, cv in sorted(result.critical_values.items()):
            rows.append({
                "": f"Critical value ({int(alpha*100)}%)",
                result.test_name: f"{cv:.4f}",
            })
        rows.append({
            "": "Reject H0 (5%)",
            result.test_name: "Yes" if result.reject_null.get(0.05, False) else "No",
        })
        rows.append({"": "N entities", result.test_name: str(result.n_entities)})
        rows.append({"": "T periods", result.test_name: str(result.n_periods)})

        df = pd.DataFrame(rows).set_index("")
        title = (
            f"Panel Unit Root Test: {result.test_name}\n"
            "H0: All entities have a unit root.\n"
            "*** p<0.01, ** p<0.05, * p<0.10"
        )
        return cls(df, title=title)

    @classmethod
    def from_multiple_results(cls, results: List, title: str = "") -> "UnitRootTable":
        """Build a comparison table from multiple UnitRootResult objects."""
        rows = []
        test_names = [r.test_name for r in results]

        # Test statistics
        stat_row = {"": "Test statistic"}
        pval_row = {"": "p-value"}
        reject_row = {"": "Reject H0 (5%)"}
        for r in results:
            stat_row[r.test_name] = f"{r.statistic:.4f}"
            pval_row[r.test_name] = f"{r.pvalue:.4f}{_stars(r.pvalue)}"
            reject_row[r.test_name] = "Yes ✓" if r.reject_null.get(0.05, False) else "No ✗"

        rows.extend([stat_row, pval_row, reject_row])

        # Critical values
        for alpha in [0.01, 0.05, 0.10]:
            cv_row = {"": f"CV ({int(alpha*100)}%)"}
            for r in results:
                cv_row[r.test_name] = f"{r.critical_values.get(alpha, np.nan):.4f}"
            rows.append(cv_row)

        meta_row = {"": "N / T"}
        for r in results:
            meta_row[r.test_name] = f"{r.n_entities} / {r.n_periods}"
        rows.append(meta_row)

        df = pd.DataFrame(rows).set_index("")
        if not title:
            title = (
                "Panel Unit Root Tests — Comparison\n"
                "H0: All entities have a unit root.\n"
                "*** p<0.01, ** p<0.05, * p<0.10"
            )
        return cls(df, title=title)

    def render(self) -> str:
        """Render as formatted string."""
        if HAS_RICH:
            return self._render_rich()
        return self._render_tabulate()

    def _render_rich(self) -> str:
        console = Console(record=True, width=120)
        table = RichTable(
            title=self.title,
            title_style="bold cyan",
            border_style="bright_blue",
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("", style="bold", min_width=20)
        for col in self.df.columns:
            table.add_column(col, justify="center", min_width=14)

        for idx, row in self.df.iterrows():
            vals = [str(idx)]
            for v in row.values:
                s = str(v)
                if "Yes" in s:
                    vals.append(f"[bold green]{s}[/]")
                elif "No" in s:
                    vals.append(f"[red]{s}[/]")
                elif "***" in s:
                    vals.append(f"[bold green]{s}[/]")
                elif "**" in s:
                    vals.append(f"[green]{s}[/]")
                elif "*" in s:
                    vals.append(f"[yellow]{s}[/]")
                else:
                    vals.append(s)
            table.add_row(*vals)
        console.print(table)
        return console.export_text()

    def _render_tabulate(self) -> str:
        header = f"\n{'═' * 80}\n{self.title}\n{'═' * 80}\n"
        body = tabulate(self.df.reset_index(), headers="keys",
                        tablefmt="fancy_grid", showindex=False)
        return header + body + "\n"

    def to_latex(self) -> str:
        ncols = len(self.df.columns)
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Panel Unit Root Tests}",
            r"\begin{tabular}{l" + "c" * ncols + "}",
            r"\hline\hline",
        ]
        lines.append(" & ".join([""] + list(self.df.columns)) + r" \\")
        lines.append(r"\hline")
        for idx, row in self.df.iterrows():
            vals = [str(idx)] + [str(v) for v in row.values]
            lines.append(" & ".join(vals) + r" \\")
        lines.extend([r"\hline\hline", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Break Detection Table (Paper 1 style)
# ══════════════════════════════════════════════════════════════════════════════

class BreakTable:
    """
    Table for structural break detection results.

    Modeled after Table 7 in Bada et al. (2021).
    """

    def __init__(self, df: pd.DataFrame, title: str = ""):
        self.df = df
        self.title = title

    @classmethod
    def from_break_result(cls, result) -> "BreakTable":
        rows = []
        for p in sorted(result.n_breaks.keys()):
            name = result.regressor_names[p] if p < len(result.regressor_names) else f"x{p+1}"
            breaks = result.break_locations[p]
            intervals = result.stability_intervals[p]
            coefs = result.coefficients[p]

            rows.append({
                "Regressor": name,
                "# Breaks": result.n_breaks[p],
                "Break locations": ", ".join(str(b) for b in breaks) if breaks else "—",
                "Intervals": " | ".join(f"[{s},{e})" for s, e in intervals),
                "Coefficients": " | ".join(f"{c:.4f}" for c in coefs),
            })

        df = pd.DataFrame(rows)
        title = (
            f"SAW Structural Break Detection Results\n"
            f"N={result.n_entities}, T={result.n_periods} | "
            f"Threshold: {result.threshold:.4f}\n"
            f"Total breaks detected: {result.total_breaks()}"
        )
        return cls(df, title=title)

    def render(self) -> str:
        if HAS_RICH:
            return self._render_rich()
        return self._render_tabulate()

    def _render_rich(self) -> str:
        console = Console(record=True, width=130)
        table = RichTable(
            title=self.title, title_style="bold cyan",
            border_style="bright_blue", header_style="bold magenta",
            show_lines=True,
        )
        for col in self.df.columns:
            table.add_column(col, justify="center" if col != "Regressor" else "left")
        for _, row in self.df.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)
        return console.export_text()

    def _render_tabulate(self) -> str:
        header = f"\n{'═' * 90}\n{self.title}\n{'═' * 90}\n"
        body = tabulate(self.df, headers="keys", tablefmt="fancy_grid", showindex=False)
        return header + body + "\n"

    def to_latex(self) -> str:
        return self.df.to_latex(index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Simulation Table (Papers 3, 5 style)
# ══════════════════════════════════════════════════════════════════════════════

class SimulationTable:
    """
    Table for Monte Carlo size/power simulation results.

    Modeled after Tables 1–6 in Almasri et al. (2016) and
    Tables 3–6 in Li & Shukur (2013).
    """

    def __init__(self, df: pd.DataFrame, title: str = ""):
        self.df = df
        self.title = title

    @classmethod
    def from_simulation(
        cls,
        results: Dict[str, Dict[str, float]],
        title: str = "Monte Carlo Simulation Results",
    ) -> "SimulationTable":
        """
        Build from a dict of results.

        Parameters
        ----------
        results : dict
            ``{test_name: {scenario_label: rejection_rate, ...}, ...}``
        """
        df = pd.DataFrame(results)
        return cls(df, title=title)

    def render(self) -> str:
        if HAS_RICH:
            console = Console(record=True, width=120)
            table = RichTable(
                title=self.title, title_style="bold cyan",
                border_style="bright_blue", header_style="bold magenta",
                show_lines=True,
            )
            table.add_column("Scenario", style="bold")
            for col in self.df.columns:
                table.add_column(col, justify="center")
            for idx, row in self.df.iterrows():
                vals = [str(idx)]
                for v in row.values:
                    s = f"{v:.4f}" if isinstance(v, float) else str(v)
                    vals.append(s)
                table.add_row(*vals)
            console.print(table)
            return console.export_text()
        header = f"\n{'═' * 80}\n{self.title}\n{'═' * 80}\n"
        body = tabulate(self.df.reset_index(), headers="keys",
                        tablefmt="fancy_grid", showindex=False)
        return header + body + "\n"

    def to_latex(self) -> str:
        return self.df.to_latex()
