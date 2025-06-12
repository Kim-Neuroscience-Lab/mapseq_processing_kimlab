# MAPseq Analysis Script
MAPseq processing code based on previous works and designed to be used with the CSHL python pipeline.

Code found here is generally a work in progress until publication.

## **Before you run:**
- Be sure that you have processed your fastq files using the [CSHL mapseq-processing Python Pipeline](https://github.com/ZadorLaboratory/mapseq-processing).
- A preprocessing and data aggregation script is provided to prepare a individaul and combined cohort level dataframe for analysis using the per-animal sample.nbcm.tsv files produced by the CSHL pipeline. This script requires the user to match the nbcm header labels to their own sample labels to ensure all the data is correctly aligned when concatenated.
- This script **process-nbcm-tsv.py** uses the aggregated_cleaned_matrix.tsv produced by the preprocessing and aggregation script (or individual sample.nbcm.tsv files from the CSHL pipeline if you do not have replicates). If you want to run a full analysis, you will need to ensure that the fastq processing parameters in the CSHL script have included: your samples, your negative control, and your injection columns in the output. Partial analysis is also possible at your discretion; there is a provided truncated "sample dataset" and associated "labels" which you can check out for guidance. You will need to check the arguments for each script if these requirements are unclear.
- If you are on Windows and want to **try the GUI Wizard**, then please download the most recent **setup_wizard.exe** from the releases page. Running this will automatically install the software necessary to run all the scripts, and will create a MAPseq_Wizard.exe in the installation directory that will provide a GUI for the main **process-nbcm-tsv.py** script. You will still need to preprocess in the terminal at the moment.
- Else, from the terminal you need to setup a new conda environment, repos, and dependencies as shown below.
- Run preprocessing then main analysis scripts.
<br/>

## EXE Installation (Windows Only)

1. Download the most recent Setup_Wizard.exe from the [releases page](https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab/releases).

2. Run the file and wait for it to complete the installation (Default location is the user directory).

## CLI Installation

1. Install mini-conda for your operating system. [mini-conda quick command line install](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

2.  With conda installed create a new environment preloaded with pip

 ```
conda create -n mapseq_processing python==3.9 pip
```

3. Activate your new environment

```
conda activate mapseq_processing
```

4. Install additional repositories

```
conda config --add channels conda-forge
conda config --add channels bioconda
```

5. Browse to your git directory and clone this project

```
cd /home/your_user/git/
git clone https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab.git
```

6. Browse into the project directory and install dependencies

```
cd /mapseq_processing_kimlab/
pip install -r requirements.txt
```

7. Run the script on your preprocess_and_aggregate.tsv on a per-group basis where the input directory contains some number of nbcm.tsv files from the CSHL pipeline. (new steps not yet reflected in sample datasets)

```
python preprocess_and_aggregate.py -i /home/mwjacobs/git/mapseq_processing_jacobs/predata/adults/ -o /home/mwjacobs/git/mapseq_processing_jacobs/data/adults/
```

8. Run the main analysis script on your sample.nbcm.tsv (command below shown using included sample dataset, but typically you would run using your aggregated data from the prior step)

```
python process-nbcm-tsv.py -o /home/mwjacobs/git/mapseq_processing_jacobs/jr0375_out/ -s JR0375 -d /home/mwjacobs/git/mapseq_processing_jacobs/sample_data/JR0375.nbcm.tsv -u 2 -l "RSP,PM,AM,A,RL,AL,LM,neg,inj"
```

<br/>

## **REQUIRED Arguments**

**-o** = path to your output directory

**-d** = path to your sample.nbcm.tsv which was produced by the [CSHL mapseq-processing Python Pipeline](https://github.com/ZadorLaboratory/mapseq-processing) Or your group.aggregate.tsv file.

**-s** = prefix for your saved files

**-l** = A list of your human readable column names in the tsv (Example:"area,area,area,neg,area,inj") 
- Your list must use 'neg' for any columns containing negative controls and 'inj' for any injection site column.
- Your list can use whatever names you want for the target areas but avoid spaces and characters.
- The code will try to sort target areas if you have repeat values (visp1,visp2,visp3,audp1,audp2...).
- I do not know if you can use more than one neg and and inj in a matrix. My data does not look like that and I havent tested.
- Use the exact format shown, no spaces between your list or the code will error.

<br/>
<br/>

## **OPTIONAL Arguments**

**-u** = Changes the threshold filter for target area UMI counts where very small values (noise) will be set to zero. (default: 2) You may want to set this to the maximum value seen in your negative control as was done in Han et. al. 2018..

```
For example the default setting is 2 meaning that for every rown in your matrix the following logic will be applied

some_row_[0,1,0,35,12,1,0,120,1,0]

will be filtered with the default value of 2 to

some_row_[0,0,0,35,12,0,0,120,0,0].

Used for potential noise reduction of single UMI values in targets, but you can change this if you would like.
```

**-i** = Sets a threshold value for filtering barcodes by minimim injection site UMI. (default: 1) Han et. al. sets this to 300, Klingler et. al. 2018 sets this to 100, you may set it to your desired value.

**-t** = Sets a threshold value for filtering barcodes by requiring a minimim UMI value in at least one target area or the barcode is removed. (default: 10) Han et. al. sets this to 10, you may set it to your desired value.

**-r** = Minimum fold-difference between 'inj' value and the highest target count. Rows not meeting this threshold are removed. (default: 10) Han et. al. sets this to 10, you may set it to your desired value.

**-f** = Enable outlier filtering of barcodes. Where any target value in a row is greater than the mean of all target values in the dataset plus two standard deviations, drop that barcode. We include this argument for microdissections which neighbor the injection site and there is no good way to know if very large UMI counts are from some kind of contamination. Use this at your own discretion.

**-a** = Value for alpha. This is the signifigance threshold (default 0.05) for Bonferroni correction, False Discovery Rate correction, and the Binomial Test.

<br/>

## **BUGS**
There are a few bugs presently. 

1. The plots may experience formatting issues

2. The variables order_full and order_partial are not dynamically defined and their use is not currently implemented correctly. You may see these strings in the cli output, but they can be ignored.

<br/>

## **Old Arguments not yet removed**

**-A** = Label from your labels to match for the first "important area" (e.g., 'AL') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

**-B** = Label from your labels to match for the second "important area" (e.g., 'PM') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

