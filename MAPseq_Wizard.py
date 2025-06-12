import PySimpleGUI as sg
import subprocess
import os

layout = [
    [sg.Text("Sample Naming Prefix:"), sg.Input(key="sample_name")],
    [sg.Text("Select your nbcm.tsv (individual or aggregated):"), sg.Input(key="data_file"), sg.FileBrowse(file_types=(("CSV/TSV Files", "*.csv *.tsv"),))],
    [sg.Text("Output Directory (will not prompt on overwrite):"), sg.Input(key="out_dir"), sg.FolderBrowse()],
    [sg.Text("Alpha (see readme):"), sg.Input(key="alpha", default_text="0.05")],
    [sg.Text("Labels (see sample data for example):"), sg.Input(key="labels")],
    [sg.Text("Filter: Minimum Injection site UMI"), sg.Input(key="injection_umi_min", default_text="1")],
    [sg.Text("Filter: At least one target UMI > X"), sg.Input(key="min_target_count", default_text="10")],
    [sg.Text("Filter: Min Injection-to-Target Ratio:"), sg.Input(key="min_body_to_target_ratio", default_text="10")],
    [sg.Text("Filter: Noise. Zero any matrix value less than X"), sg.Input(key="target_umi_min", default_text="2")],
    [sg.Checkbox("Experimental: Remove high-UMI outliers where value was > (mean+2*StdDev)", key="apply_outlier_filtering")],
    [sg.Button("Run"), sg.Exit()]
]

window = sg.Window("NBCM Processing GUI Wizard", layout)

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == "Run":
        cmd = [
            "conda", "run", "-n", "mapseq_processing", "python", "process-nbcm-tsv.py",
            "--sample_name", values["sample_name"],
            "--data_file", values["data_file"],
            "--out_dir", values["out_dir"],
            "--alpha", values["alpha"],
            "--labels", values["labels"],
            "--injection_umi_min", values["injection_umi_min"],
            "--min_target_count", values["min_target_count"],
            "--min_body_to_target_ratio", values["min_body_to_target_ratio"],
            "--target_umi_min", values["target_umi_min"]
        ]

        if values["apply_outlier_filtering"]:
            cmd += ["--apply_outlier_filtering"]

        print("🔧 Running:", " ".join(cmd))
        subprocess.run(cmd)

window.close()
