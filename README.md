# mapseq_processing_jacobs
MAPseq processing code based on previous works and designed to be used with the CSHL python pipeline.

Notebooks are for my records only, not kept up to date at the moment.
Any code found here is a work in progress with no guarantee that it will work until otherwise stated.

The most functional code is the process_ncbm.py script. Edit this script before running.
	You need to convert your nbcm.tsv to a .csv and add a single line header in the first row.
	In the header label each column "your target area names"
	Edit the script replacing "rsp, pm, am, a, rl, al, lm, cere" with your column labels. You will need to alter some matrix layout logic to help the script understand how many sample types you have.
 Edit the out_dir, save_name, and data_file values in the script.

 Install

 To install you need to setup a conda environment and the listed dependencies. All should be available in bioconda, conda-forge, and the default.

 Then just run the script and watch the output. If you get the upsetplots out then you got all the way through.
