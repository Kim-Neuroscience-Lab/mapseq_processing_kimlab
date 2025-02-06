# mapseq_processing_jacobs
MAPseq processing code based on previous works and designed to be used with the CSHL python pipeline.

Any code found here is a work in progress with no guarantee that it will do what it claims until otherwise stated.

**Before you run:**
 Setup a conda environment and the listed dependencies. All should be available in bioconda, conda-forge, and the default.

Run --help to see arguments.

**REQUIRED Arguments**

**-0** = path to your output directory

**-d** = path to your nbcm.tsv

**-s** = prefix for your saved files

**-l** = list of your columns in the tsv (Example:"area,area,area,neg,area,inj") You must use 'neg' for any columns containing negative controls and 'inj' for any injection site column. You can use whatever names you want for samples but avoid spaces and characters. The code will try to sort samples if you have repeat values (visp1,visp2,visp3,audp1,audp2...). I do not know if you can use more than one neg and and inj in a matrix. My data does not look like that.

**OPTIONAL Arguments**

**-f** = Enable outlier filtering using mean + 2*std deviation. Removes barcodes where a value in the row is >= to the mean+2*stddev.

**alpha** Signifigance threshold (default 0.05) for Bonferroni correction, False Discovery Rate correction, and the Binomial Test.

**target_umi_min** = filter for low counts in the matrix eg some_row_[0,1,0,35,12,1,0,120,1,0] will be filtered with the default value of 2 to some_row_[0,0,35,12,0,0,120,0,0].

If you get the upsetplots out in your analysis folder then you got all the way through.

**BUGS**
There are a few bugs presently. 

1.The plots are not all in a format that I love.

2. order_partial is not dynamically defined and is not currently implemented correctly.

**Example command for running the sample data**

python process-nbcm-tsv.py -f -o /home/mwjacobs/git/mapseq_processing_jacobs/jr0375_out/ -s JR0375 -d /home/mwjacobs/git/mapseq_processing_jacobs/sample_data/JR0375.nbcm.tsv -u 2 -l "RSP,PM,AM,A,RL,AL,LM,neg,inj"

**Old Arguments not yet removed**

**-A** = Label from your labels to match for the first important area (e.g., 'AL') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

**-B** = Label from your labels to match for the second important area (e.g., 'PM') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

