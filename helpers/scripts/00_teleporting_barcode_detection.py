import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"


REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "00_cleaned_data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "teleporting_barcode_detection"
DEFAULT_FILE_GLOBS = ["*.nbcm.tsv", "*.nbcm.all.tsv"]
UNMAPPED_BATCH = "UNMAPPED"
UMI_RANK_THRESHOLD = 2

# Hardcoded sequencing experiment -> animal mapping provided by user.
# Matching is case-insensitive by normalizing animal IDs to upper-case.
SEQUENCING_BATCH_PAIRS = [
    ("M265", "M759"),
    ("M265", "M760"),
    ("M265", "M777"),
    ("M275", "JR0375"),
    ("M275", "JR0376"),
    ("M277", "JR0422"),
    ("M277", "JR0420"),
    ("M277", "JR0448"),
    ("M277", "JR0446"),
    ("M292", "JR0671"),
    ("M292", "JR0547"),
    ("M300", "M759"),
    ("M300", "M760"),
    ("M300", "M761"),
    ("M300", "JR0670"),
    ("M300", "JR0686"),
    ("M300", "JR0672"),
    ("M300", "JR0674"),
    ("M300", "JR0678"),
    ("M300", "JR0552"),
    ("M300", "JR0694"),
    ("M300", "JR0692"),
    ("M300", "JR0695"),
    ("M300", "JR0548"),
    ("M312", "JR0884"),
    ("M312", "JR0883"),
    ("M312", "JR0887"),
    ("M312", "JR0689"),
    ("M312", "JR0690"),
    ("M312", "JR0685"),
    ("M312", "JR0546"),
    ("M312", "JR0693"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect teleporting barcodes (barcodes observed in multiple animal matrices) "
            "from cleaned MAPseq data."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with age-group folders and animal matrices (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--file_glob",
        action="append",
        default=None,
        help=(
            "Filename glob to include (can be specified multiple times). "
            "Defaults to: *.nbcm.tsv and *.nbcm.all.tsv"
        ),
    )
    return parser.parse_args()


def discover_matrix_files(input_dir: Path, file_globs):
    discovered = set()
    for pattern in file_globs:
        discovered.update(input_dir.rglob(pattern))
    files = sorted([p for p in discovered if p.is_file()])
    if not files:
        raise FileNotFoundError(
            f"No matrix files found in {input_dir} using globs: {', '.join(file_globs)}"
        )
    return files


def extract_animal_id(file_path: Path):
    name = file_path.name
    for suffix in [".nbcm.all.tsv", ".nbcm.tsv"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return file_path.stem


def extract_age_group(input_dir: Path, file_path: Path):
    rel = file_path.relative_to(input_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "unknown"


def age_sort_key(age_label: str):
    if age_label.lower().startswith("p") and age_label[1:].isdigit():
        return (0, int(age_label[1:]))
    return (1, age_label.lower())


def build_animal_to_batches():
    mapping = defaultdict(set)
    for batch_id, animal_id in SEQUENCING_BATCH_PAIRS:
        mapping[animal_id.upper()].add(batch_id)
    return {k: sorted(v) for k, v in mapping.items()}


def load_animal_barcodes(input_dir: Path, matrix_files, animal_to_batches):
    records = []
    for file_path in matrix_files:
        age = extract_age_group(input_dir, file_path)
        animal_id = extract_animal_id(file_path)
        animal_label = f"{age}:{animal_id}"
        batches = animal_to_batches.get(animal_id.upper(), [])
        batch_label = ";".join(batches) if batches else UNMAPPED_BATCH

        df = pd.read_csv(file_path, sep="\t", dtype=str)
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in matrix file: {file_path}")

        barcode_series = df.iloc[:, 0].fillna("").astype(str).str.strip()
        barcode_series = barcode_series[barcode_series != ""]
        barcode_set = set(barcode_series.tolist())

        records.append(
            {
                "age": age,
                "animal_id": animal_id,
                "animal_label": animal_label,
                "sequencing_batch": batch_label,
                "file_path": str(file_path),
                "n_rows": int(len(df)),
                "n_unique_barcodes": int(len(barcode_set)),
                "barcodes": barcode_set,
            }
        )
    return records


def build_barcode_index(animal_records):
    barcode_to_animals = defaultdict(set)
    for rec in animal_records:
        for barcode in rec["barcodes"]:
            barcode_to_animals[barcode].add(rec["animal_label"])
    return barcode_to_animals


def build_teleporting_table(teleporting_map, animal_to_age, animal_to_batch_label=None, include_batch_columns=False):
    rows = []
    for barcode, animals in teleporting_map.items():
        animals_sorted = sorted(animals)
        ages_sorted = sorted({animal_to_age[a] for a in animals_sorted}, key=age_sort_key)
        row = {
            "barcode": barcode,
            "n_animals": len(animals_sorted),
            "animals": ";".join(animals_sorted),
            "ages": ";".join(ages_sorted),
        }
        if include_batch_columns and animal_to_batch_label is not None:
            batches = set()
            for animal in animals_sorted:
                label = animal_to_batch_label.get(animal, UNMAPPED_BATCH)
                for batch in label.split(";"):
                    if batch:
                        batches.add(batch)
            batches_sorted = sorted(batches)
            row["n_batches"] = len(batches_sorted)
            row["batches"] = ";".join(batches_sorted)
        rows.append(row)
    teleport_df = pd.DataFrame(rows)
    if not teleport_df.empty:
        teleport_df = teleport_df.sort_values(
            by=["n_animals", "barcode"], ascending=[False, True]
        ).reset_index(drop=True)
    return teleport_df


def build_animal_summary(animal_records, teleporting_map):
    teleporting_set = set(teleporting_map.keys())
    rows = []
    for rec in animal_records:
        total = rec["n_unique_barcodes"]
        teleporting_count = sum(1 for b in rec["barcodes"] if b in teleporting_set)
        unique_count = total - teleporting_count
        unique_ratio = (unique_count / total) if total > 0 else 0.0
        unique_percent = unique_ratio * 100.0

        rows.append(
            {
                "age": rec["age"],
                "animal_id": rec["animal_id"],
                "animal_label": rec["animal_label"],
                "sequencing_batch": rec["sequencing_batch"],
                "file_path": rec["file_path"],
                "total_barcodes": total,
                "teleporting_barcodes_in_animal": teleporting_count,
                "unique_barcodes": unique_count,
                "unique_ratio": unique_ratio,
                "unique_percent": unique_percent,
                "teleporting_percent": 100.0 - unique_percent,
                "duplicate_rows_within_matrix": rec["n_rows"] - rec["n_unique_barcodes"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["age_sort"] = summary_df["age"].apply(age_sort_key)
    summary_df = summary_df.sort_values(
        by=["age_sort", "animal_id"], ascending=[True, True]
    ).drop(columns=["age_sort"])
    return summary_df.reset_index(drop=True)


def write_run_summary(
    output_path: Path,
    animal_summary_df: pd.DataFrame,
    barcode_index,
    teleporting_df,
    summary_title: str,
):
    total_animals = len(animal_summary_df)
    total_unique_global = len(barcode_index)
    teleporting_count = len(teleporting_df)
    teleporting_global_ratio = (
        teleporting_count / total_unique_global if total_unique_global else 0.0
    )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{summary_title}\n")
        handle.write("=" * len(summary_title) + "\n")
        handle.write(f"Animals scanned: {total_animals}\n")
        handle.write(f"Global distinct barcodes: {total_unique_global}\n")
        handle.write(f"Teleporting barcodes (>=2 animals): {teleporting_count}\n")
        handle.write(f"Teleporting fraction (global): {teleporting_global_ratio:.8f}\n")
        if "sequencing_batch" in animal_summary_df.columns:
            n_unmapped = int((animal_summary_df["sequencing_batch"] == UNMAPPED_BATCH).sum())
            handle.write(f"Animals without mapped sequencing batch: {n_unmapped}\n")
        handle.write("\nPer-animal uniqueness\n")
        handle.write("---------------------\n")
        handle.write(
            f"Mean unique percent: {animal_summary_df['unique_percent'].mean():.8f}\n"
        )
        handle.write(
            f"Median unique percent: {animal_summary_df['unique_percent'].median():.8f}\n"
        )
        handle.write(
            f"Minimum unique percent: {animal_summary_df['unique_percent'].min():.8f}\n"
        )
        handle.write(
            f"Maximum unique percent: {animal_summary_df['unique_percent'].max():.8f}\n"
        )
        handle.write("\nTop animals by teleporting barcode count\n")
        handle.write("----------------------------------------\n")
        top_rows = animal_summary_df.sort_values(
            by=["teleporting_barcodes_in_animal", "animal_label"], ascending=[False, True]
        ).head(10)
        for _, row in top_rows.iterrows():
            handle.write(
                f"{row['animal_label']}: teleporting={int(row['teleporting_barcodes_in_animal'])}, "
                f"total={int(row['total_barcodes'])}, unique_percent={row['unique_percent']:.8f}\n"
            )


def plot_metric(summary_df: pd.DataFrame, metric_col: str, ylabel: str, title: str, output_prefix: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(summary_df) * 0.45), 4))
    ax.bar(summary_df["animal_label"], summary_df[metric_col], color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    fig.savefig(str(output_prefix.with_suffix(".svg")))
    plt.close(fig)


def build_batch_aggregate_summary(animal_records):
    batch_to_records = defaultdict(list)
    for rec in animal_records:
        for batch_id in rec["sequencing_batch"].split(";"):
            batch_id = batch_id.strip()
            if batch_id:
                batch_to_records[batch_id].append(rec)

    rows = []
    for batch_id in sorted(batch_to_records.keys()):
        records = batch_to_records[batch_id]
        barcode_to_animals_within_batch = defaultdict(set)
        for rec in records:
            for barcode in rec["barcodes"]:
                barcode_to_animals_within_batch[barcode].add(rec["animal_label"])

        teleporting_unique_count = sum(
            1 for animals in barcode_to_animals_within_batch.values() if len(animals) >= 2
        )
        total_barcodes_sum = sum(rec["n_unique_barcodes"] for rec in records)
        teleporting_ratio = (
            teleporting_unique_count / total_barcodes_sum if total_barcodes_sum > 0 else 0.0
        )
        unique_percent = 100.0 * (1.0 - teleporting_ratio)

        rows.append(
            {
                "sequencing_batch": batch_id,
                "n_animals": len(records),
                "total_barcodes_sum": int(total_barcodes_sum),
                "teleporting_unique_count": int(teleporting_unique_count),
                "teleporting_ratio": float(teleporting_ratio),
                "unique_percent": float(unique_percent),
            }
        )

    return pd.DataFrame(rows)


def plot_batch_metric(summary_df: pd.DataFrame, metric_col: str, ylabel: str, title: str, output_prefix: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary_df["sequencing_batch"], summary_df[metric_col], color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Sequencing Batch")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    fig.savefig(str(output_prefix.with_suffix(".svg")))
    plt.close(fig)


def build_global_batch_aggregate_summary(batch_aggregate_df: pd.DataFrame, teleporting_by_batch_df: pd.DataFrame):
    n_batches = int(len(batch_aggregate_df))
    n_animals = int(batch_aggregate_df["n_animals"].sum()) if not batch_aggregate_df.empty else 0
    global_total_barcodes_sum = (
        int(batch_aggregate_df["total_barcodes_sum"].sum()) if not batch_aggregate_df.empty else 0
    )
    global_teleporting_unique_count = int(len(teleporting_by_batch_df))
    global_teleporting_ratio = (
        global_teleporting_unique_count / global_total_barcodes_sum
        if global_total_barcodes_sum > 0
        else 0.0
    )
    global_unique_percent = 100.0 * (1.0 - global_teleporting_ratio)

    return pd.DataFrame(
        [
            {
                "n_batches": n_batches,
                "n_animals": n_animals,
                "global_total_barcodes_sum": global_total_barcodes_sum,
                "global_teleporting_unique_count": global_teleporting_unique_count,
                "global_teleporting_ratio": float(global_teleporting_ratio),
                "global_unique_percent": float(global_unique_percent),
            }
        ]
    )


def build_global_umi_rank_curve_data(matrix_files, umi_threshold=UMI_RANK_THRESHOLD):
    barcode_to_global_umi = defaultdict(float)

    for file_path in matrix_files:
        df = pd.read_csv(file_path, sep="\t")
        if df.shape[1] < 2:
            continue

        barcode_series = df.iloc[:, 0].fillna("").astype(str).str.strip()
        numeric_block = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        row_umi = numeric_block.sum(axis=1)

        per_file = pd.DataFrame({"barcode": barcode_series, "umi": row_umi})
        per_file = per_file[per_file["barcode"] != ""]
        per_file = per_file.groupby("barcode", as_index=False)["umi"].sum()

        for _, row in per_file.iterrows():
            barcode_to_global_umi[row["barcode"]] += float(row["umi"])

    rank_df = pd.DataFrame(
        [{"barcode": barcode, "global_total_umi": umi} for barcode, umi in barcode_to_global_umi.items()]
    )
    total_barcodes_before_filter = int(len(rank_df))
    if rank_df.empty:
        rank_df = pd.DataFrame(columns=["rank", "barcode", "global_total_umi"])
        return rank_df, total_barcodes_before_filter, 0

    rank_df = rank_df[rank_df["global_total_umi"] >= float(umi_threshold)].copy()
    rank_df = rank_df.sort_values(by="global_total_umi", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
    total_barcodes_after_filter = int(len(rank_df))
    return rank_df, total_barcodes_before_filter, total_barcodes_after_filter


def plot_global_umi_rank_curve(rank_df: pd.DataFrame, output_prefix: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rank_df["rank"], rank_df["global_total_umi"], linewidth=1.2, color="#4C78A8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Barcode Rank (descending UMI)")
    ax.set_ylabel("Global Total UMI")
    ax.set_title("Global UMI Rank-Abundance Curve")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    fig.savefig(str(output_prefix.with_suffix(".svg")))
    plt.close(fig)


def plot_global_umi_rank_curve_log10_y_linear_x(rank_df: pd.DataFrame, output_prefix: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    y_log10 = np.log10(rank_df["global_total_umi"].astype(float))
    ax.plot(rank_df["rank"], y_log10, linewidth=1.2, color="#4C78A8")
    ax.set_xlabel("Barcode Rank (descending UMI)")
    ax.set_ylabel("log10(read_count)")
    ax.set_title("Global UMI Rank-Abundance Curve (linear rank, log10 abundance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    fig.savefig(str(output_prefix.with_suffix(".svg")))
    plt.close(fig)


def add_normalized_fraction_columns(rank_df: pd.DataFrame):
    normalized_df = rank_df.copy()
    if normalized_df.empty:
        normalized_df["umi_fraction"] = []
        normalized_df["umi_cumulative_fraction"] = []
        normalized_df["umi_remaining_fraction"] = []
        return normalized_df

    total_umi = float(normalized_df["global_total_umi"].sum())
    if total_umi <= 0:
        normalized_df["umi_fraction"] = 0.0
        normalized_df["umi_cumulative_fraction"] = 0.0
        normalized_df["umi_remaining_fraction"] = 0.0
        return normalized_df

    normalized_df["umi_fraction"] = normalized_df["global_total_umi"] / total_umi
    normalized_df["umi_cumulative_fraction"] = normalized_df["umi_fraction"].cumsum()
    normalized_df["umi_remaining_fraction"] = 1.0 - normalized_df["umi_cumulative_fraction"]
    return normalized_df


def plot_normalized_rank_curve(
    normalized_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_prefix: Path,
    invert_x_axis=False,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(normalized_df["rank"], normalized_df[y_col], linewidth=1.2, color="#4C78A8")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    if invert_x_axis:
        ax.invert_xaxis()
    ax.set_xlabel("Barcode Rank (descending UMI)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    fig.savefig(str(output_prefix.with_suffix(".svg")))
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    file_globs = args.file_glob if args.file_glob else DEFAULT_FILE_GLOBS

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_files = discover_matrix_files(input_dir, file_globs)
    animal_to_batches = build_animal_to_batches()
    animal_records = load_animal_barcodes(input_dir, matrix_files, animal_to_batches)
    barcode_index = build_barcode_index(animal_records)

    animal_to_batch_label = {
        rec["animal_label"]: rec["sequencing_batch"] for rec in animal_records
    }
    teleporting_map = {b: animals for b, animals in barcode_index.items() if len(animals) >= 2}
    teleporting_by_batch_map = {}
    for barcode, animals in barcode_index.items():
        batches = set()
        for animal in animals:
            label = animal_to_batch_label.get(animal, UNMAPPED_BATCH)
            batches.update([x for x in label.split(";") if x])
        if len(batches) >= 2:
            teleporting_by_batch_map[barcode] = animals

    animal_to_age = {rec["animal_label"]: rec["age"] for rec in animal_records}

    teleporting_df = build_teleporting_table(teleporting_map, animal_to_age)
    teleporting_by_batch_df = build_teleporting_table(
        teleporting_by_batch_map,
        animal_to_age,
        animal_to_batch_label=animal_to_batch_label,
        include_batch_columns=True,
    )
    animal_summary_df = build_animal_summary(animal_records, teleporting_map)
    animal_summary_by_batch_df = build_animal_summary(animal_records, teleporting_by_batch_map)

    teleporting_csv = output_dir / "teleporting_barcodes.csv"
    teleporting_by_batch_csv = output_dir / "teleporting_barcodes_by_batch.csv"
    animal_summary_csv = output_dir / "animal_uniqueness_summary.csv"
    animal_summary_by_batch_csv = output_dir / "animal_uniqueness_summary_by_batch.csv"
    batch_uniqueness_summary_csv = output_dir / "batch_uniqueness_summary.csv"
    run_summary_txt = output_dir / "run_summary.txt"
    run_summary_by_batch_txt = output_dir / "run_summary_by_batch.txt"
    run_summary_batch_aggregate_txt = output_dir / "run_summary_batch_aggregate.txt"
    global_batch_aggregate_csv = output_dir / "global_batch_aggregate_summary.csv"
    global_batch_aggregate_txt = output_dir / "global_batch_aggregate_summary.txt"
    global_umi_rank_curve_csv = output_dir / "global_umi_rank_curve_data.csv"
    global_umi_rank_curve_normalized_csv = output_dir / "global_umi_rank_curve_normalized_data.csv"
    global_umi_rank_curve_summary_txt = output_dir / "global_umi_rank_curve_summary.txt"

    teleporting_df.to_csv(teleporting_csv, index=False)
    teleporting_by_batch_df.to_csv(teleporting_by_batch_csv, index=False)
    animal_summary_df.to_csv(animal_summary_csv, index=False)
    animal_summary_by_batch_df.to_csv(animal_summary_by_batch_csv, index=False)
    batch_aggregate_df = build_batch_aggregate_summary(animal_records)
    batch_aggregate_df.to_csv(batch_uniqueness_summary_csv, index=False)
    global_batch_aggregate_df = build_global_batch_aggregate_summary(
        batch_aggregate_df, teleporting_by_batch_df
    )
    global_batch_aggregate_df.to_csv(global_batch_aggregate_csv, index=False)
    global_umi_rank_curve_df, umi_rank_total_before, umi_rank_total_after = build_global_umi_rank_curve_data(
        matrix_files, umi_threshold=UMI_RANK_THRESHOLD
    )
    global_umi_rank_curve_df.to_csv(global_umi_rank_curve_csv, index=False)
    global_umi_rank_curve_normalized_df = add_normalized_fraction_columns(global_umi_rank_curve_df)
    global_umi_rank_curve_normalized_df.to_csv(global_umi_rank_curve_normalized_csv, index=False)
    write_run_summary(
        run_summary_txt,
        animal_summary_df,
        barcode_index,
        teleporting_df,
        summary_title="Teleporting barcode detection summary (global)",
    )
    write_run_summary(
        run_summary_by_batch_txt,
        animal_summary_by_batch_df,
        barcode_index,
        teleporting_by_batch_df,
        summary_title="Teleporting barcode detection summary (sequencing-batch-aware)",
    )
    with run_summary_batch_aggregate_txt.open("w", encoding="utf-8") as handle:
        handle.write("Batch aggregate uniqueness summary (intra-batch only)\n")
        handle.write("=====================================================\n")
        handle.write("Each batch is analyzed independently; no cross-batch comparisons.\n")
        handle.write(
            "unique_percent = 100 * (1 - teleporting_unique_count / total_barcodes_sum)\n\n"
        )
        for _, row in batch_aggregate_df.iterrows():
            handle.write(
                f"{row['sequencing_batch']}: "
                f"n_animals={int(row['n_animals'])}, "
                f"total_barcodes_sum={int(row['total_barcodes_sum'])}, "
                f"teleporting_unique_count={int(row['teleporting_unique_count'])}, "
                f"unique_percent={row['unique_percent']:.8f}\n"
            )
    with global_batch_aggregate_txt.open("w", encoding="utf-8") as handle:
        row = global_batch_aggregate_df.iloc[0]
        handle.write("Global aggregate summary across sequencing experiments\n")
        handle.write("=====================================================\n")
        handle.write(
            "global_unique_percent = 100 * (1 - global_teleporting_unique_count / global_total_barcodes_sum)\n\n"
        )
        handle.write(f"n_batches={int(row['n_batches'])}\n")
        handle.write(f"n_animals={int(row['n_animals'])}\n")
        handle.write(f"global_total_barcodes_sum={int(row['global_total_barcodes_sum'])}\n")
        handle.write(
            f"global_teleporting_unique_count={int(row['global_teleporting_unique_count'])}\n"
        )
        handle.write(f"global_teleporting_ratio={row['global_teleporting_ratio']:.10f}\n")
        handle.write(f"global_unique_percent={row['global_unique_percent']:.8f}\n")
    with global_umi_rank_curve_summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("Global UMI rank-abundance summary\n")
        handle.write("=================================\n")
        handle.write(f"UMI threshold for tail trimming: {UMI_RANK_THRESHOLD}\n")
        handle.write(f"Total barcodes before threshold: {umi_rank_total_before}\n")
        handle.write(f"Total barcodes after threshold: {umi_rank_total_after}\n")
        if umi_rank_total_after > 0:
            handle.write(
                f"Max global UMI (rank 1): {global_umi_rank_curve_df['global_total_umi'].iloc[0]:.6f}\n"
            )
            handle.write(
                f"Min global UMI (last rank): {global_umi_rank_curve_df['global_total_umi'].iloc[-1]:.6f}\n"
            )
            handle.write(
                f"Sum umi_fraction: {global_umi_rank_curve_normalized_df['umi_fraction'].sum():.10f}\n"
            )
            handle.write(
                "Final cumulative fraction: "
                f"{global_umi_rank_curve_normalized_df['umi_cumulative_fraction'].iloc[-1]:.10f}\n"
            )
            handle.write(
                "First remaining fraction: "
                f"{global_umi_rank_curve_normalized_df['umi_remaining_fraction'].iloc[0]:.10f}\n"
            )
            handle.write(
                "Last remaining fraction: "
                f"{global_umi_rank_curve_normalized_df['umi_remaining_fraction'].iloc[-1]:.10f}\n"
            )
            handle.write(
                "Also wrote global_umi_rank_curve_log10_y_linear_x.png/.svg (linear rank x, log10 read_count y).\n"
            )

    plot_metric(
        animal_summary_df,
        metric_col="unique_percent",
        ylabel="Unique Barcodes (%)",
        title="Unique Barcode Percentage by Animal",
        output_prefix=output_dir / "unique_percent_by_animal",
    )
    plot_metric(
        animal_summary_df,
        metric_col="teleporting_percent",
        ylabel="Teleporting Barcodes (%)",
        title="Teleporting Barcode Percentage by Animal",
        output_prefix=output_dir / "teleporting_percent_by_animal",
    )
    plot_metric(
        animal_summary_by_batch_df,
        metric_col="unique_percent",
        ylabel="Unique Barcodes (%)",
        title="Unique Barcode Percentage by Animal (Sequencing-Batch-Aware)",
        output_prefix=output_dir / "unique_percent_by_animal_by_batch",
    )
    plot_metric(
        animal_summary_by_batch_df,
        metric_col="teleporting_percent",
        ylabel="Teleporting Barcodes (%)",
        title="Teleporting Barcode Percentage by Animal (Sequencing-Batch-Aware)",
        output_prefix=output_dir / "teleporting_percent_by_animal_by_batch",
    )
    plot_batch_metric(
        batch_aggregate_df,
        metric_col="unique_percent",
        ylabel="Unique Barcodes (%)",
        title="Unique Barcode Percentage by Sequencing Batch (Intra-Batch)",
        output_prefix=output_dir / "unique_percent_by_sequencing_batch",
    )
    plot_batch_metric(
        batch_aggregate_df,
        metric_col="teleporting_ratio",
        ylabel="Teleporting Ratio",
        title="Teleporting Barcode Ratio by Sequencing Batch (Intra-Batch)",
        output_prefix=output_dir / "teleporting_ratio_by_sequencing_batch",
    )
    if umi_rank_total_after > 0:
        plot_global_umi_rank_curve(
            global_umi_rank_curve_df, output_prefix=output_dir / "global_umi_rank_curve"
        )
        plot_global_umi_rank_curve_log10_y_linear_x(
            global_umi_rank_curve_df,
            output_prefix=output_dir / "global_umi_rank_curve_log10_y_linear_x",
        )
        plot_normalized_rank_curve(
            global_umi_rank_curve_normalized_df,
            y_col="umi_fraction",
            y_label="UMI Fraction of Global Total",
            title="Global UMI Rank Fraction Curve",
            output_prefix=output_dir / "global_umi_rank_fraction_curve",
        )
        plot_normalized_rank_curve(
            global_umi_rank_curve_normalized_df,
            y_col="umi_cumulative_fraction",
            y_label="Cumulative UMI Fraction of Global Total",
            title="Global UMI Rank Cumulative Fraction Curve",
            output_prefix=output_dir / "global_umi_rank_cumulative_fraction_curve",
        )
        plot_normalized_rank_curve(
            global_umi_rank_curve_normalized_df,
            y_col="umi_remaining_fraction",
            y_label="Remaining UMI Fraction of Global Total",
            title="Global UMI Rank Remaining Fraction Curve",
            output_prefix=output_dir / "global_umi_rank_remaining_fraction_curve",
        )
        plot_normalized_rank_curve(
            global_umi_rank_curve_normalized_df,
            y_col="umi_cumulative_fraction",
            y_label="Cumulative UMI Fraction of Global Total",
            title="Global UMI Rank Cumulative Fraction Curve (Reversed X)",
            output_prefix=output_dir / "global_umi_rank_cumulative_fraction_curve_reversed_x",
            invert_x_axis=True,
        )

    print(f"Scanned input directory: {input_dir}")
    print(f"Matrix files found: {len(matrix_files)}")
    print(f"Animals analyzed: {len(animal_records)}")
    print(f"Global distinct barcodes: {len(barcode_index)}")
    print(f"Teleporting barcodes (global): {len(teleporting_df)}")
    print(f"Teleporting barcodes (sequencing-batch-aware): {len(teleporting_by_batch_df)}")
    print(f"Wrote teleporting table: {teleporting_csv}")
    print(f"Wrote sequencing-batch-aware teleporting table: {teleporting_by_batch_csv}")
    print(f"Wrote animal summary: {animal_summary_csv}")
    print(f"Wrote sequencing-batch-aware animal summary: {animal_summary_by_batch_csv}")
    print(f"Wrote batch aggregate summary: {batch_uniqueness_summary_csv}")
    print(f"Wrote global batch aggregate summary: {global_batch_aggregate_csv}")
    print(f"Wrote run summary: {run_summary_txt}")
    print(f"Wrote sequencing-batch-aware run summary: {run_summary_by_batch_txt}")
    print(f"Wrote batch aggregate run summary: {run_summary_batch_aggregate_txt}")
    print(f"Wrote global batch aggregate text summary: {global_batch_aggregate_txt}")
    print(f"Wrote global UMI rank curve data: {global_umi_rank_curve_csv}")
    if umi_rank_total_after > 0:
        print(
            f"Wrote global UMI rank curve (log10 y, linear rank x): "
            f"{output_dir / 'global_umi_rank_curve_log10_y_linear_x.png'}"
        )
    print(f"Wrote global UMI normalized rank data: {global_umi_rank_curve_normalized_csv}")
    print(f"Wrote global UMI rank curve summary: {global_umi_rank_curve_summary_txt}")
    print(f"Global UMI rank points kept (UMI >= {UMI_RANK_THRESHOLD}): {umi_rank_total_after}")
    print(
        "Global aggregate unique percent across sequencing experiments: "
        f"{global_batch_aggregate_df.iloc[0]['global_unique_percent']:.8f}"
    )
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
