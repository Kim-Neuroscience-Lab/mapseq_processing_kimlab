#!/usr/bin/env python3
"""
Transform long-format individual replicate CSVs to wide format for Prism.

Helper 01 writes two files per model:
- individual_replicates_per_animal_global.csv (normalized_freq: each animal sums to 1)
- individual_replicates_per_animal_domain.csv (domain_normalized_freq: each domain sums to 1; jr0420 NaN)

Use --input to point at either file; default is _global.

Output: tab-separated file with row 1 = timepoints repeated per replicate, column 1 = motif,
cells = chosen value column; row order = PRISM_ROW_ORDER (singles LM,AL,AM,PM,RSP then 2- then 3-region).
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


# Default timepoint order (use data order if not overridden)
DEFAULT_TIMEPOINT_ORDER = ["P12", "P20", "P3", "P60"]

# Exact row order for Prism: singles (LM, AL, AM, PM, RSP), then 2-region, then 3-region.
PRISM_ROW_ORDER = [
    "LM",
    "AL",
    "AM",
    "PM",
    "RSP",
    '"AL\', \'AM"',
    '"AL\', \'LM"',
    '"AL\', \'PM"',
    '"AL\', \'RSP"',
    '"AM\', \'LM"',
    '"AM\', \'PM"',
    '"AM\', \'RSP"',
    '"LM\', \'PM"',
    '"LM\', \'RSP"',
    '"PM\', \'RSP"',
    '"AL\', \'AM\', \'LM"',
    '"AL\', \'AM\', \'PM"',
    '"AL\', \'AM\', \'RSP"',
    '"AL\', \'LM\', \'PM"',
    '"AL\', \'LM\', \'RSP"',
    '"AL\', \'PM\', \'RSP"',
    '"AM\', \'LM\', \'PM"',
    '"AM\', \'LM\', \'RSP"',
    '"AM\', \'PM\', \'RSP"',
    '"LM\', \'PM\', \'RSP"',
]


def motif_to_display(motif: str) -> str:
    """Convert motif to Prism display format: UPPERCASE single; tuple-style for compounds."""
    if not motif or (isinstance(motif, float) and pd.isna(motif)):
        return ""
    s = str(motif).strip()
    if not s:
        return ""
    if "+" not in s:
        return s.upper()
    parts = sorted(p.strip().upper() for p in s.split("+") if p.strip())
    if not parts:
        return ""
    # Tuple-style as in reference: "AL', 'AM" -> so format is "PART1', 'PART2', 'PART3"
    return '"' + "', '".join(parts) + '"'


def motif_order_key(motif: str) -> tuple:
    """Order by domain (number of regions) then alphabetically."""
    if not motif or (isinstance(motif, float) and pd.isna(motif)):
        return (0, "")
    s = str(motif).strip()
    domain = 1 + s.count("+") if s else 0
    return (domain, s)


def write_wide_for_prism(
    input_path,
    output_path=None,
    value_column="normalized_freq",
    timepoint_order=None,
):
    """
    Read long-format individual_replicates_per_animal CSV and write wide-format TSV for Prism.
    Can be called from helper 01 after writing the long CSVs.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path = Path(output_path) if output_path else input_path.parent / (input_path.stem + "_wide_for_prism.tsv")
    timepoint_order = timepoint_order or DEFAULT_TIMEPOINT_ORDER

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["Motif"])
    df = df[df["Motif"].astype(str).str.strip() != ""]
    if df.empty:
        raise ValueError("No rows with valid Motif.")
    if value_column not in df.columns:
        raise ValueError(f"Column {value_column} not in CSV.")

    df["MotifDisplay"] = df["Motif"].apply(motif_to_display)
    df = df[df["MotifDisplay"] != ""]

    wide = df.pivot_table(
        index="MotifDisplay",
        columns=["Timepoint", "Animal_ID"],
        values=value_column,
        aggfunc="first",
    )
    tp_order = {t: i for i, t in enumerate(timepoint_order)}
    cols_list = list(wide.columns)
    cols_list.sort(key=lambda c: (tp_order.get(c[0], 999), str(c[1])))
    wide = wide[cols_list]

    order_index = {label: i for i, label in enumerate(PRISM_ROW_ORDER)}
    row_order = sorted(
        wide.index.tolist(),
        key=lambda m: order_index.get(m, len(PRISM_ROW_ORDER)),
    )
    wide = wide.reindex(row_order)

    with open(output_path, "w") as f:
        header_cells = [""] + [str(c[0]) for c in wide.columns]
        f.write("\t".join(header_cells) + "\n")
        for idx in wide.index:
            row_vals = [str(idx)]
            for c in wide.columns:
                v = wide.loc[idx, c]
                row_vals.append("" if pd.isna(v) else str(v))
            f.write("\t".join(row_vals) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Transform long individual_replicates_per_animal.csv to wide Prism format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input CSV path (long format). Use .../individual_replicates_per_animal_global.csv or _domain.csv.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output TSV path. Default: same dir as input, suffix _wide_for_prism.tsv",
    )
    parser.add_argument(
        "--value_column",
        choices=["normalized_freq", "domain_normalized_freq"],
        default="normalized_freq",
        help="Which column to use for values (default: normalized_freq).",
    )
    parser.add_argument(
        "--timepoints",
        type=str,
        default=None,
        help="Comma-separated timepoint order for columns, e.g. P12,P20,P60 to drop P3. Default: P12,P20,P3,P60.",
    )
    args = parser.parse_args()

    if args.input is None:
        args.input = Path(__file__).parent.parent.parent / "02_output/05.HAN_filter_parameters_i300_r10_t10_u5_helpers/01_motif_analysis_per_animal/uniform/individual_replicates_per_animal_global.csv"
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    timepoint_order = DEFAULT_TIMEPOINT_ORDER
    if args.timepoints:
        timepoint_order = [t.strip() for t in args.timepoints.split(",") if t.strip()]

    try:
        output_path = write_wide_for_prism(
            args.input,
            output_path=args.output,
            value_column=args.value_column,
            timepoint_order=timepoint_order,
        )
        print(f"Wrote wide Prism format: {output_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
