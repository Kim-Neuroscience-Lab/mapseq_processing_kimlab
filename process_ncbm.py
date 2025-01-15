import os
import sympy
import numpy as np
import pandas as pd
from sympy import symbols, Product, Array, N
from sympy.printing import latex
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib import patches
import seaborn as sn
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from scipy.stats import friedmanchisquare, kruskal, binomtest
from sklearn.cluster import k_means
from sklearn.preprocessing import normalize 
import itertools
from adjustText import adjust_text
import multiprocessing as mp
import itertools
import upsetplot as up

### User Configuration

# Directories and Save Name
out_dir = "/mnt/d/mapseq/test/output/"
save_name = "P3_combined"
data_file = r'/mnt/d/mapseq/test/data/p3_nbcm_combined_for-preprocess - Sheet1.csv'  ##This is your nbcm (individuals are provided by CSHL python code output. A combined file could be made by hand."

#This switch is for excluding some columns. See like 254
full_data = True
motif_join = '+'

### Helper Functions

def calculate_projections_from_matrix(matrix):
    """Calculate projections as the count of non-zero values in each column."""
    column_counts = {region: np.count_nonzero(matrix[:, idx]) for idx, region in enumerate([
        'RSP', 'PM', 'AM', 'A', 'RL', 'AL', 'LM', 'Cerebellum'])}
    # Remove the last column (Cerebellum) if not needed
    column_counts.pop('Cerebellum', None)
    return column_counts

def calculate_total_projections(projections):
    return sum(projections.values())

def solve_for_roots(projections, observed_cells):
    N0, k = symbols('N_0 k')
    m = len(projections) - 1
    s = Array(list(projections.values()))
    pi = (1 - Product((1 - (s[k]/N0)), (k, 0, m)).doit())
    soln = sympy.solve(pi * N0 - observed_cells)
    roots = [N(x).as_real_imag()[0] for x in soln]
    return roots, pi

def save_latex_expression(expression, title, filename):
    latex_output = latex(expression)
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"${latex_output}$", fontsize=14, va='center', ha='center', wrap=True)
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def calculate_probabilities(projections, total_projections):
    return {region: (count / total_projections) for region, count in projections.items()}

def binomial_test(value, total, probability):
    return binomtest(value, n=total, p=probability).pvalue

def normalize_rows(matrix):
    """Normalize rows of a matrix by their maximum value."""
    return np.apply_along_axis(lambda x: x / np.amax(x) if np.amax(x) > 0 else x, axis=1, arr=matrix)

def clean_and_filter(matrix):
    """Perform cleaning and filtering on the matrix."""
    # Remove rows with all zero values
    matrix = matrix[np.any(matrix, axis=1)]
    
    # Remove rows with all zero target values (columns 0-6)
    matrix = matrix[np.any(matrix[:, :-1], axis=1)]

    # Remove rows where column 7 (DLS) has positive values
    matrix = matrix[matrix[:, 7] <= 0]

    # Apply a threshold filter for minimum counts in target columns (columns 0-6)
    min_threshold = 2
    matrix = matrix[np.amax(matrix[:, :-1], axis=1) >= min_threshold]

    return matrix

### Main Calculations

# Load barcodes
barcodematrix = np.genfromtxt(data_file, delimiter=',')
barcodematrix = np.array(barcodematrix, dtype=np.float64)
print("Barcode Matrix Shape:", barcodematrix.shape)

# Perform cleaning and filtering
filtered_matrix = clean_and_filter(barcodematrix)
print("Filtered Matrix Shape:", filtered_matrix.shape)

# Normalize rows of the filtered matrix
normalized_matrix = normalize_rows(filtered_matrix)
print("Normalized Matrix Shape:", normalized_matrix.shape)

# Drop the last column (Cerebellum)
normalized_matrix = np.delete(normalized_matrix, 7, axis=1)

# Save normalized matrix to CSV with headers
columns = ["RSP", "PM", "AM", "A", "RL", "AL", "LM"]
normalized_matrix_file = os.path.join(out_dir, f"{save_name}_Normalized_Matrix.csv")
pd.DataFrame(normalized_matrix, columns=columns).to_csv(normalized_matrix_file, index=False, float_format="%.8f")
print(f"Normalized matrix saved to: {normalized_matrix_file}")

# Calculate observed cells dynamically
observed_cells = filtered_matrix.shape[0]  # Number of rows after filtering
print(f"Observed Cells: {observed_cells}")

# Calculate projections dynamically from the filtered matrix
projections = calculate_projections_from_matrix(filtered_matrix)
print("Projections:", projections)

# Calculate total projections
total_projections = calculate_total_projections(projections)
print(f"Total Projections: {total_projections}")

# Solve for roots and simplify Pi
roots, pi = solve_for_roots(projections, observed_cells)
print("Roots:", roots)

simplified_pi = sympy.simplify(pi)
print("Simplified Pi:", simplified_pi)

# Save LaTeX representation of simplified Pi
save_latex_expression(simplified_pi, "Simplified Pi Visualization", os.path.join(out_dir, f"{save_name}_Simplified_Pi.png"))

# Calculate probabilities
psdict = calculate_probabilities(projections, total_projections)
print("Region-specific Probabilities:", psdict)

# Solve for p_e
pe = symbols('p_e')
#pe = pe_num
solution = sympy.solve((1 - (1 - pe)**len(projections)) * total_projections - observed_cells, [pe], force=True)
if solution:
    pe_num = float(solution[0])
    print("Derived p_e:", pe_num)
else:
    raise ValueError("No valid solution for p_e found.")

# Numerical Calculations
scaled_value = psdict['RSP'] * psdict['PM']
calculated_value = (1 - (1 - pe_num)**len(projections)) * total_projections
print("Calculated Value:", calculated_value)

# Save LaTeX representation of calculated value
save_latex_expression(calculated_value, "Calculated Value Visualization", os.path.join(out_dir, f"{save_name}_Calculated_Value.png"))

# Perform statistical tests
std_dev = np.sqrt(scaled_value * total_projections * (1 - scaled_value))
print("Standard Deviation:", std_dev)

binomial_p_value = binomial_test(94, total_projections, scaled_value)
print("Binomial Test Result (P-Value):", binomial_p_value)

# Save results
results = {
    "Roots": roots,
    "Simplified Pi": [simplified_pi],
    "Region-specific Probabilities": list(psdict.values()),
    "Calculated Value": [calculated_value],
    "Standard Deviation": [std_dev],
    "Binomial Test Result (P-Value)": [binomial_p_value]
}
os.makedirs(out_dir, exist_ok=True)
for key, value in results.items():
    pd.DataFrame({key: value}).to_csv(os.path.join(out_dir, f"{save_name}_{key.replace(' ', '_')}.csv"), index=False)

### Visualizations
# Region-specific probabilities
plt.figure(figsize=(8, 5))
plt.bar(psdict.keys(), psdict.values())
plt.title("Region-specific Probabilities")
plt.ylabel("Probability")
plt.xlabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{save_name}_Region_Probabilities.png"))
plt.close()

# Roots scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(range(len(roots)), roots)
plt.title("Roots")
plt.ylabel("Root Value")
plt.xlabel("Index")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{save_name}_Roots.png"))
plt.close()

### Analysis and Plotting Integration

#Where is the normalized_matrix.csv
data_dir = out_dir
file_name = os.path.join(f"{save_name}_Normalized_Matrix.csv")


# Load normalized matrix as input for analysis
normalized_matrix_file = os.path.join(out_dir, f"{save_name}_Normalized_Matrix.csv")
normalized_matrix = pd.read_csv(normalized_matrix_file)

# Ensure 'analysis' subdirectory exists within 'out_dir'
analysis_dir = os.path.join(out_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

#Where do you want the analysis output to go?
plot_dir = analysis_dir

n0 = observed_cells #import from stats at beginning

np.set_printoptions(suppress=True)


def load_df(file, remove_cols=['RSP'], subset=['PM','AM','A','RL','AL','LM']):
    """
    Loads excel file specified in file which is a string for the full path
    Excel should have column names as the first row (header)
    remove_col specifies the names of columns to remove (list, e.g. ["DLS"])
    subset specifies the names of columns to keep (drop others, e.g. 'LH', 'BLA', 'PFC', 'NAc'])
    """
    experiment_ = pd.read_csv(file,header=0) #pd.read_excel(file,header=0)
    print(experiment_.columns)
    df = experiment_
    #df.columns = colnames
    if remove_cols is not None:
        try:
            df = df.drop(columns=remove_cols)
        except Exception as e:
            print("!!! Error: Could not remove columns. Column not found in remove_cols")
    if subset is not None:
        try:
            df = df[subset] #limit, removes BNST and CeA
        except Exception as e:
            print("!!! Error: Could not subset columns. Column not found in subset")
    return df

"""
Change the file path:
"""
file_path = data_dir + file_name
if full_data:
    #Load full data set
    df = load_df(file_path, remove_cols=None,subset=None)
else:
    #Load special regions:
    df = load_df(file_path, remove_cols=['RSP'], subset=['PM','AM','A','RL','AL','LM'])

print("df shape: {}".format(df.shape))
print("DF Head:")
print(df.head())
print("Number of NAs:")
print(df.isnull().sum())

# Find optimal number of clusters
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = k_means(df.to_numpy(), n_clusters=k)
    #km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km[2])
len(Sum_of_squared_distances)

elbow_plt = plt.figure(figsize=(10,7))
s1 = np.array(Sum_of_squared_distances[0:-1])
s2 = np.array(Sum_of_squared_distances[1:])
plt.plot(K[0:-1], np.abs(s2-s1), 'x-', color='red',label='Delta Inertia')
plt.plot(K, Sum_of_squared_distances, 'x-', color='blue',label='Inertia')
plt.xlabel('k',fontsize=20)
plt.ylabel('Inertia',fontsize=20)
plt.title('Elbow Method For Optimal k',fontsize=20)
plt.legend()
#Looks like optimal cluster number is 6

elbow_plt.savefig(os.path.join(plot_dir, save_name + "elbow_plot.pdf"))

km = k_means(df.to_numpy(), n_clusters=6) #THIS WAS INITIALLY 6, I THINK THEY PULL IT FROM BOX 1065 OUTPUT average below.
km[0].shape

df.to_csv(os.path.join(plot_dir, save_name + "_motif_obs_exp.csv"))

scolors = ['black','red','orange','yellow'] #['lightblue','darkblue'] 
scm = LinearSegmentedColormap.from_list(
        'white_to_red', scolors, N=100)
fig,ax = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(10,10)
clusters,regions = km[0].shape
ax.set_title("K-means Clustering")
ax.set_xlabel("Regions")
ax.set_ylabel("Cluster")
ax.set_xticks(range(regions))
ax.set_xticklabels(df.columns.to_list())
##
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks(range(0,clusters,1))
ax.set_yticklabels(range(1,clusters+1,1))
X = range(regions)
for i in range(km[0].shape[0]):
    y = np.array([i]).repeat(regions)
    #km[0][i]
    size = km[0][i]
    size = (size - size.min()) / (size.max() - size.min())
    ax_ = ax.scatter(x=X,y=y,s=1000,cmap=scm,c=size)

fig.colorbar(ax_,label='Projection Strength')

fig.savefig(os.path.join(plot_dir, save_name + "_kmeans.pdf"))

def concatenate_list_data(slist,join=motif_join):
    result = []
    for i in slist:
        sub = ''
        for j in i:
            if sub:
                sub = sub + join + str(j)
            else:
                sub += str(j)
        result.append(sub)
    return result

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def gen_motifs(r,labels): #r is number of regions, so number of motifis is 2^r
    num_motifs = 2**r
    motifs = np.zeros((num_motifs,r)).astype(bool)
    motif_ids = list(powerset(np.arange(r)))
    motif_labels = [] #list of labels e.g. PFC-LH or PFC-LS-BNST
    for i in range(num_motifs):
        idx = motif_ids[i]
        motifs[i,idx] = True
        #motif_labels.append(labels[[idx]].to_list())
        label = labels[np.array(idx)].to_list() if idx else ['']
        motif_labels.append(label)
    return motifs, motif_labels

def count_motifs(df,motifs,return_ids=False):
    """
    Returns a vector with counts that indicated number of motifs present for each possible motif
    A motif is a combination of target areas represented as a binary array, 
    e.g. [1,1,0] represents a motif where a cell targets the first two regions but not the 3rd
    A motif can be obtained by simply thresholding each cell's projection strength vector such that
    non-zero elements are 1.
    Also returns the labels for each motif
    """
    cells, regions = df.shape
    data = df.to_numpy().astype(bool)
    #np.unique(df.to_numpy().astype(bool),axis=0,return_counts=True)
    #if motifs is None: motifs = gen_motifs(regions)
    counts = np.zeros(motifs.shape[0])
    cell_ids = []
    for i in range(motifs.shape[0]): #loop through motifs (128x7)
        cell_ids_ = []
        for j in range(data.shape[0]): #loop through observed data cells X regions
            if np.array_equal(motifs[i],data[j]):
                counts[i] = counts[i] + 1
                cell_ids_.append(j)
        cell_ids.append(cell_ids_)
        
    if return_ids:
        return counts, cell_ids#, motifs
    else:
        return counts

def zip_with_scalar(l, o):
    return zip(l, itertools.repeat(o))

motifs, motif_labels = gen_motifs(df.shape[1],df.columns)

dcounts, cell_ids = count_motifs(df,motifs, return_ids=True) #observed data

def convert_counts_to_df(columns,counts,labels):
    """
    Deprecated. Use get_obs_exp(...) instead.
    
    Returns dataframe showing cell counts for each motif
    """
    motifdf = pd.DataFrame(columns=columns)
    for i in range(len(counts)):
        #other = {motif_labels[i][j]:1 for j in range(len(motif_labels[i]))}
        cols = labels[i]
        if len(cols) == 1 and not cols[0]:
            continue
        motifdf.loc[i,cols] = 1
        motifdf.loc[i,"Count"] = counts[i]
    return motifdf.fillna(0).infer_objects(copy=False)

motif_df = convert_counts_to_df(df.columns,dcounts,motif_labels)

from sympy import N

def get_expected_counts(motifs, num_regions = 7, prob_edge=pe_num,n=n0):
    # Ensure variables are numeric
    prob_edge = float(prob_edge.evalf()) if hasattr(prob_edge, "evalf") else float(prob_edge)
    #motif_set = set(['PFC', 'NAc', 'LS', 'BNST', 'LH', 'BA', 'CeA'])
    n_motifs = len(motifs)
    res = np.zeros(n_motifs)
    probs = np.zeros(n_motifs)
    for i,motif in enumerate(motifs):
        e1 = int(len(motif))
        e2 = num_regions - e1
        p = (prob_edge ** e1) * (1 - prob_edge) ** e2
        exp = float(N(p)) * n
        res[i] = exp
        probs[i] = p
    res[0] = 0
    return res, probs

exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels),\
                                dcounts,\
                                exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif','Observed','Expected']
df_obs_exp.to_csv(os.path.join(plot_dir, save_name + "_motif_obs_exp.csv"))
df_obs_exp

##CHATGPT suggested addition to give a csv without the null combination from the powerset.

exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels), dcounts, exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif', 'Observed', 'Expected']

# Exclude empty motifs
df_obs_exp = df_obs_exp[df_obs_exp['Motif'] != ""]  # Adjust condition as needed for your data format - chatGPT

# Save filtered data to CSV
df_obs_exp.to_csv(
    os.path.join(plot_dir, save_name + "_motif_obs_exp_filtered.csv"), 
    index=False
)


def standardize_pos(x):
    return (x + 1) / (x.std())
def standardize(x):
    return (x + 1e-13) / (x.max() - x.min())
def subset_list(lis, ids):
    return [lis[i] for i in ids]

def get_motif_sig_pts(dcounts,labels,\
                            prob_edge=pe_num, n0 = n0, \
                      exclude_zeros=True, \
                      p_transform=lambda x: -1 * np.log10(x)):
    #dcounts: motif counts from observed data
    num_motifs = dcounts.shape[0]
    expected, probs = get_expected_counts(labels, prob_edge=pe_num,n=n0)
    assert dcounts.shape[0] == expected.shape[0]
    if exclude_zeros:
        nonzid = np.nonzero(dcounts)[0]
    else:
        nonzid = np.arange(dcounts.shape[0])
    num_nonzid_motifs = nonzid.shape[0]
    dcounts_ = dcounts[nonzid]
    expected_ = expected[nonzid]
    probs_ = probs[nonzid]
    #Effect size is log2(observed/expected)
    effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
    matches = np.zeros(num_nonzid_motifs)
    assert dcounts_.shape[0] == expected_.shape[0]
    dcounts_ = dcounts_.astype(int)
    for i in range(num_nonzid_motifs):
        #pi = probs_[i] if probs_[i] > 0 else 1.0e-99
        #dcounts_[i] = int(dcounts_[i])  # Force integer conversion
        pi = max(probs_[i], 1e-10) #avoid zero or very small probs
        matches[i] = binomtest(int(dcounts_[i]),n=n0,p=pi).pvalue #alternative='greater'     ########IF I REMOVE THIS AND TRY
        #matches[i] = matches[i] if matches[i] > 0 else 1.0e-10
        matches[i] = max(matches[i], 1e-10)
    matches = p_transform(matches)
    #matches is the significance level
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

sigs, slabels = get_motif_sig_pts(dcounts,motif_labels,exclude_zeros=True)

#Bonferroni correction: p-threshold / Num comparisons
pcutoff = -1*np.log10(0.05 / len(slabels)) # = 3.1 for thresh of 0.05 or 3.6 for thresh of 0.01 for n=63

list_sig = [i for (i,(e,s)) in enumerate(sigs) if s > pcutoff ]
color_labels = ['gray' for i in range(len(sigs))]
for i in list_sig:
    e,s = sigs[i]
    if e > 0: #overrepresented
        color_labels[i] = 'red'
    else:
        color_labels[i] = 'blue'
#color_labels

hide_singlets = True
if hide_singlets:
    mask = [i for (i,l) in enumerate(slabels) if len(l) > 1]
#subset_list(slabels,[1,5,7])
fig,ax = plt.subplots(1,1)#plt.figure(figsize=(13,11))
fig.set_size_inches(20,20)
#ax.set_ylim([0,12])
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
ax.set_title(save_name.replace('_',''),fontsize=16)
ax.set_xlabel("Effect Size \n$log_2($observed/expected$)$",fontsize=16)
ax.set_ylabel("Significance\n $-log_{10}(P)$",fontsize=16)
ax.axhline(y=pcutoff, linestyle='--')
ax.axvline(x=0, linestyle='--')
ax.text(x=-.5,y=pcutoff+0.05,s='P-value cutoff',fontsize=16)
##CHATGPT UPDATES FOR FORMATTING
from adjustText import adjust_text

# Scatter plot
ax.scatter(*zip(*subset_list(sigs, mask)), c=subset_list(color_labels, mask))

# Prepare text labels
pretty_slabels = concatenate_list_data(subset_list(slabels, mask))
coordinates = subset_list(sigs, mask)
texts = []

for n, (z, y) in enumerate(coordinates):
    txt = pretty_slabels[n]
    texts.append(ax.text(z, y, txt, fontsize=12))

# Adjust y-axis limits
y_vals = [y for _, y in subset_list(sigs, mask)]
padding = 0.1 * (max(y_vals) - min(y_vals))  # Add 10% padding
ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)

# Adjust text positions to avoid overlap
adjust_text(
    texts,
    #only_move={'points': 'y', 'text': 'y'},  # Allow vertical movement
    arrowprops=dict(
        arrowstyle="->",
        color='gray',
        lw=1,
        shrinkA=1,  # These values won't matter much for adjust_text
        shrinkB=1
    ),
    expand_points=(1.5, 2.5),  # Add padding around points
    force_text=1,  # Increase separation force for text
    force_points=1  # Increase separation force for points
)

fig.savefig(os.path.join(plot_dir, save_name + "_effect_significance.pdf"))

def gen_per_cell_plot(df,cell_ids,motif_labels,dcounts,expected,savepath=plot_dir, hide_singlets=True,figsize=(16,35)):
    """
    This plots each cell of a given motif on the same plot as an individual line
    Each line's points are the corresponding projection strengths at that region
    So this plot shows the projection strengths of all the cells for each motif
    """
    if hide_singlets: #Only show motifs with two or more regions
        mask = [i for (i,l) in enumerate(motif_labels) if len(l) > 1]
        cell_ids = subset_list(cell_ids,mask)
        
    non0cell_ids = [(i,x) for (i,x) in enumerate(cell_ids) if len(x) > 0]
    dcounts = subset_list(dcounts,mask)
    exp_counts_ = subset_list(expected,mask)
    obs_ex = []
    for i,x in non0cell_ids:
        oe = (dcounts[i],exp_counts_[i])
        obs_ex.append(oe)
    num_plots = len(non0cell_ids)
    plot_titles = concatenate_list_data(subset_list(motif_labels,mask))
    #print(num_plots)
    ncols = 2
    nrows = int(np.ceil(num_plots / ncols))
    fig = plt.figure(figsize=figsize)
    n = 1
    for cellids_ in non0cell_ids:
        """if n > 2:
            break"""
        i,cellids = cellids_
        ax = fig.add_subplot(nrows,ncols,n)
        title = plot_titles[i]
        ax.set_title(title)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.to_list())
        ax.set_ylabel("Projection Strength")
        #x = df[df.index.isin(cellids)].to_numpy()
        x = df.iloc[cellids,:].to_numpy()
        ## add observed/expected legend
        obs,ex = obs_ex[n-1]
        textstr = 'Observed: {} \n Expected: {}'.format(int(obs),int(ex))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ##
        yerr = x.std(axis=0) / np.sqrt(x.shape[0])
        #ax.errorbar(x=np.arange(df.shape[1]), y=x.mean(axis=0),yerr=yerr,ecolor='black') # mean with error bars
        for j in range(x.shape[0]):
            ax.plot(np.arange(df.shape[1]), x[j], markerfacecolor='none', alpha=0.2, c='gray') 
        ax.errorbar(x=np.arange(df.shape[1]), y=x.mean(axis=0),yerr=yerr,ecolor='gray', c='black',linewidth=3) # mean with error bars
        n+=1
    if savepath:
        fig.savefig(savepath)
    return ax

#Run the function
if full_data:
    fig_size2 = (20,140) #(20,140)
else:
    fig_size2 = (20,10)
gprcpplot = gen_per_cell_plot(df,cell_ids,motif_labels,dcounts,exp_counts,figsize=fig_size2, savepath = os.path.join(plot_dir, save_name + "_per_cell_proj_strength.pdf"))
#
#

def show_perc_motifs(perc=True):
    if perc:
        return list(zip(dcounts / (dcounts.sum() / 100),motif_labels))
    else:
        return list(zip(dcounts,motif_labels))

colors = ['white','red']
#colors = ['white','blue']
'''cm = LinearSegmentedColormap.from_list(
        'white_to_red', colors, N=100)'''
cm = LinearSegmentedColormap.from_list(
        'white_to_red', colors, N=100)

from sklearn.preprocessing import StandardScaler

# Draw heatmap
sn.light_palette("#ef6e6e",n_colors=2,as_cmap=True)
order_full = reversed(['LM','AL','RL','A','AM','PM','RSP']) #add 'Cere' to this list if the neg controls are included.
order_partial = ['LM','AL','RL','AM','PM']
if full_data:
    df_ = df[order_full]
else:
    df_ = df[order_partial]

print("Number of Nulls:")
print(df_.isnull().sum())
print("Number of Nas:")
print(df_.isna().sum())
print("Number of Infs:")
print(np.isinf(df_).sum())

scaler = StandardScaler()
df_ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns)

clusterfig = sn.clustermap(df_,col_cluster=False,metric='cosine', method='average',\
              cbar_kws=dict(label='Projection Strength'), \
                cmap=cm,vmin=0.0,vmax=1.0)
clusterfig.ax_heatmap.set_title(save_name.replace('_',''))
clusterfig.ax_heatmap.axes.get_yaxis().set_visible(False)
#sn.heatmap(experiment,cbar_kws=dict(shrink=0.5,label='Projection Strength'))
#clusterfig.savefig("/Users/brandonbrown/Desktop/KheirbekLab/MAPseq/clustermap.pdf")

clusterfig.savefig(os.path.join(plot_dir, save_name + "_red_white_cluster_heatmap.pdf"))

def gen_prob_matrix(df : pd.DataFrame):
    data = df.to_numpy(copy=True)
    cells,regions = data.shape
    mat = np.zeros((regions,regions)) #area B x area A
    #loop over columns (region )
    for col in range(regions):
        #find all cells (rows in data) that project to 'col'
        ids_col = np.where(data[:,col] != 0)[0]
        sub_col = data[ids_col]
        #of these, how many project to region B
        for row in range(regions):
            ids_row = np.where(sub_col[:,row] != 0)[0]
            if ids_col.shape[0] == 0:
                prob = 0
            else:
                prob = ids_row.shape[0] / ids_col.shape[0]
            #print("P({} | {}) = {}".format(df.columns[row],df.columns[col],prob))
            mat[col,row] = prob
    mat = pd.DataFrame(mat, columns=df.columns)
    mat.index = df.columns
    return mat

probmat = gen_prob_matrix(df)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_title(save_name.replace('_',''),fontsize=20)
#colors2 = ['black','red','orange','yellow']
colors2 = ['darkblue','#1f9ed1','#26ffc5','#ffc526','yellow']
cm2 = LinearSegmentedColormap.from_list(
        'white_to_red', colors2, N=100)
ax.set_facecolor('#a8a8a8')
ax = sn.heatmap(probmat.T,mask=probmat.T == 1,ax=ax,cbar_kws=dict(label='$P(B | A)$'),cmap=cm2) #can add vmax=number for scale
ax.set_xlabel("Area A",fontsize=16)
ax.set_ylabel("Area B",fontsize=16)
plt.savefig(os.path.join(plot_dir, save_name + "_blueyellow_probability_heatmap.pdf"))

def remove_zero_rows(df):
    df_ = df.fillna(0)
    df = df.loc[~(df_==0).all(axis=1)].astype('float32')
    return df

def get_overlaps(df):
    """
    Returns the number of cells that target both regions in a pair
    """
    cells,regions = df.shape
    pairs = list(itertools.combinations(df.columns,2)) #remove null id
    pairs_unzip = list(zip(*pairs))
    from_r = list(pairs_unzip[0])
    to_r = list(pairs_unzip[1])
    counts=[]
    df = df.copy()
    for i in pairs:
        sub = df.T.loc[list(i)].T
        sub = remove_zero_rows(sub)
        counts.append(sub.shape[0])
    res = pd.DataFrame(columns=['from','to','value'])
    res['from'] = from_r
    res['to'] = to_r
    res['value'] = counts
    return res #counts, pairs

oo = get_overlaps(df)
oo.head()

def get_motif_count(motif,counts,labels):
    """
    Get the number of cells that project to this specific motif
    where motif is a list of column names e.g. ['LH','PFC']
    """
    for i in range(len(labels)):
        if set(motif) == set(labels[i]):
            return counts[i]

get_motif_count(['PM','AL'],dcounts,motif_labels)

import re
pattern = re.compile('([^\s\w]|_)+')

def strip_nonchars(string):
    strip = pattern.sub('', string)
    return strip

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def get_pair_reg_props(df,counts,labels):
    """
    Given each pair in region_list, find number of cells that target either in the pair
    and then find proportion of cells that target both in pair exclusively
    
    counts: motif counts
    labels: motif labels
    """
    region_list = df.columns.to_list()
    R = len(region_list)
    tot = df.shape[0] / 100
    pairs = findsubsets(region_list,2)
    results = []
    for i,pair in enumerate(pairs):
        p1,p2 = pair
        num_cells_p1 = df[df[p1] > 0.].shape[0]
        num_cells_p2 = df[df[p2] > 0.].shape[0]
        tot_cells = num_cells_p1 + num_cells_p2
        num_doublets = get_motif_count([p1,p2],counts,labels)
        perc = np.around((100.0 * num_doublets) / tot_cells,3)
        results.append((p1,p2,tot_cells,num_doublets,perc))
    return results

get_pair_reg_props(df,dcounts,motif_labels)

def get_all_counts(df,motifs,counts,labels):
    """
    Returns an array where each row is a motif and the counts of 
    number of cells targeting each member of the motif (non-exclusive), total number of cells targeting any of 
    the members of the motif, number of cells targeting all members of motif, and percentage exclusively targeting full
    motif (relative to any member of the motif), e.g.
    columns: PFC BNST LS CeA Total Motif Perc
    row 1  : 10   20  30  NA  60    6     10%
    where NA means that region is not part of the motif
    
    Input: df; dataframe of normalized data, Num cells (N) x Num regions (R)
    motifs M (num motifs) x R binary matrix indicating which regions present in each motif (row)
    counts vector containg counts of cells that exclusively project to each matching motif/row in motifs
    labels string labels for regions that make each matching motif in motifs
    """
    ret = pd.DataFrame(columns=df.columns.to_list() + ['Total', 'Motif Num', 'Motif Perc'])
    num_cols = len(ret.columns.to_list())
    for i,motif in enumerate(motifs): #loop through motifs
        m = [index for (index,x) in enumerate(motif) if x]
        if len(m) < 1:
            continue
        sums = df.iloc[:,m].astype(bool).astype(int).sum().to_numpy()
        ap = np.zeros(num_cols)
        ap[:] = np.nan
        ap[m] = sums
        ap = ap.reshape(1,ap.shape[0])
        ap = pd.DataFrame(ap,columns=ret.columns)
        tot = ap.iloc[:,0:-3].dropna(axis=1).to_numpy().sum()
        ap.iloc[:,-3] = tot
        ap.iloc[:,-2] = counts[i]
        # Safeguard against division by zero
        if tot == 0:
            ap.iloc[:, -1] = 0.0
        else:
            ap.iloc[:, -1] = 100.0 * (counts[i] / tot)
        ret = pd.concat([ret, ap], ignore_index=True)
    return ret

def get_all_counts_nondf(df,motifs,counts,labels):
    """
    Returns an array where each row is a motif and the columns are the counts of 
    number of cells targeting each member of the motif (non-exclusive), total number of cells targeting any of 
    the members of the motif, number of cells targeting all members of motif, and percentage exclusively targeting full
    motif (relative to any member of the motif), e.g.
    columns: PFC BNST LS CeA Total Motif Perc
    row 1  : 10   20  30  NA  60    6     10%
    where NA means that region is not part of the motif
    
    Input: df; dataframe of normalized data, Num cells (N) x Num regions (R)
    motifs M (num motifs) x R binary matrix indicating which regions present in each motif (row)
    counts vector containg counts of cells that exclusively project to each matching motif/row in motifs
    labels string labels for regions that make each matching motif in motifs
    """
    retdf = [] #return list
    #each element is a list [Labels, R1 count, R2 count ... Rn count, Total Count, Motif Count, Motif Perc]
    for i,motif in enumerate(motifs): #loop through motifs
        m = [index for (index,x) in enumerate(motif) if x]
        row = list(np.zeros(1+len(m)+3)) #1 (labels) + num-regions-in-motifs + 3 (total,motif count,motif perc)
        if len(m) < 1:
            continue
        sums = df.iloc[:,m].astype(bool).astype(int).sum().to_numpy()
        row[0] = labels[i]
        row[1:len(m)+1] = sums
        tot = sums.sum()
        
        # Prevent division by zero
        if tot == 0:
            row[len(m) + 1] = np.nan  # or 0, depending on how you want to handle this case
            row[len(m) + 2] = np.nan  # Handle the motif count
            row[len(m) + 3] = np.nan  # Handle the motif percentage
        else:
            row[len(m) + 1] = tot
            row[len(m) + 2] = counts[i]
            row[len(m) + 3] = 100.0 * (counts[i] / tot)
        
        retdf.append(row)
    return retdf

unstruct_counts = get_all_counts_nondf(df,motifs,dcounts,motif_labels)

def write_motif_counts(path,counts):
    with open(path, 'w') as f:
        for item in counts:
            f.write("%s\n" % item)

write_motif_counts(
    os.path.join(plot_dir, save_name + '_counts.txt'), 
    unstruct_counts
)

mdf = get_all_counts(df,motifs,dcounts,motif_labels)

mdf.head()

mdf.to_csv(os.path.join(plot_dir, save_name + "_motif_counts.csv"))

show_perc_motifs(False)

def get_target_pie(df : pd.DataFrame):
    """
    For each cell (row), determine how many projections it makes
    """
    data = df.to_numpy(copy=True)
    cells,regions = data.shape
    res = []#np.zeros(regions)
    for cell in range(cells):
        num_targets = int(np.nonzero(data[cell])[0].shape[0])
        res.append(num_targets)
    ret = np.array(res)
    #ret = pd.DataFrame(ret)
    return ret

df_pie = get_target_pie(df)

g,c = np.unique(df_pie,return_counts=True)

c_row_names = ['1 target']
c_row_names += ["{} targets".format(i+2) for i in range(c.shape[0]-1)]
c = pd.DataFrame(c,columns=['# Cells'], index=c_row_names)
c_np = c.to_numpy(copy=True).flatten()
c.head()

c.to_csv(os.path.join(plot_dir, save_name + "_pie_chart_data.csv"))

c_tot = c_np.sum()
c_tot

plt.figure(figsize=(10,10))
plt.title(save_name.replace('_',''))
glabels = ["1 target \n {:0.3}\%".format(100*c_np[0] / c_tot)]
glabels += ["{} targets \n {:0.3}\%".format(i+2,100*j/c_tot) for (i,j) in zip(range(c_np.shape[0]-1),c_np[1:])]
patches, texts = plt.pie(c.to_numpy().flatten(),labels=glabels)
[txt.set_fontsize(8) for txt in texts]
plt.savefig(os.path.join(plot_dir, save_name + "_num_targets_pie.pdf"))

maxproj = TSNE(n_components=2,metric='cosine').fit_transform(df.to_numpy(copy=True))

#maxprojclusters = k_means(X=maxproj,n_clusters=6)

tlabels = df.to_numpy(copy=True).argmax(axis=1)
#tlabels = km[1]

plt.figure(figsize=(12,9))
plt.title(save_name.replace('_',''),fontsize=20)
plt.xlabel("tSNE Component 1",fontsize=20)
plt.ylabel("tSNE Component 2",fontsize=20)
sc = plt.scatter(maxproj[:,0],maxproj[:,1],c=tlabels) #c=maxprojclusters[1]
cb = plt.colorbar(sc)
cb.set_label("Maximum Projection Target",fontsize=20)
plt.savefig(os.path.join(plot_dir, save_name + "_tsne.pdf"))

def prepare_upset_data(df):
    #mask1 = [i for (i,x) in enumerate(motif_labels) if len(x) > 1]
    mask1 = [i for (i,x) in enumerate(df['Degree'].to_list()) if x > 1]
    a = subset_list(df['Motifs'].to_list(), mask1)
    b = df['Observed'][mask1]
    c = df['Expected'][mask1]
    d = df['Expected SD'][mask1]
    e = df['Effect Size'][mask1]
    f = df['P-value'][mask1]
    g = df['Group'][mask1]
    mask2 = [i for i in range(b.shape[0]) if b.iloc[i] > 0]
    a = subset_list(a, mask2)
    b = b.iloc[mask2]
    b = b.to_numpy().astype(int)
    #
    c = c.iloc[mask2]
    c = c.to_numpy().astype(int)
    #
    d = d.iloc[mask2]
    #
    e = e.iloc[mask2]
    #
    f = f.iloc[mask2]
    #
    g = g.iloc[mask2]
    dfdata = pd.DataFrame(data=[a,b,c,d,e,f,g]).T
    dfdata.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Group']
    #dfdata = dfdata.sort_values(by="Observed",ascending=False)
    return dfdata

sigsraw, slabelsraw = get_motif_sig_pts(dcounts,motif_labels,exclude_zeros=False, p_transform=lambda x:x)

effectsigsraw = np.array(sigsraw)
expected_sd_raw = np.array([np.sqrt(motif_probs[i] * n0 * (1-motif_probs[i])) for i in range(len(slabelsraw))])

degree = [len(x) for x in motif_labels]
degree[0] = 0

group = []
bonferroni_correction = len(slabels)
for i in range(len(degree)):
    """
    Group 1: motifs significantly over represented
    Group 2: motifs non-sig over-represented
    Group 3: motifs non-sig under-represented
    Group 4: motifs significantly under represented
    """
    grp = 0
    thr = 0.05 / bonferroni_correction
    if effectsigsraw[i,0] > 0: #over-represented
        if effectsigsraw[i,1] < thr: #statistically significant
            grp = 1
        else:
            grp = 3 #2
    else: #under-represented
        if effectsigsraw[i,1] > thr:
            grp = 4
        else: #statistically significant
            grp = 2
    group.append(grp)

dfraw = pd.DataFrame(data=[
                           motif_labels,\
                           dcounts,exp_counts.astype(int), \
                          expected_sd_raw,effectsigsraw[:,0], effectsigsraw[:,1], degree, group]).T
dfraw.columns=['Motifs','Observed','Expected', 'Expected SD','Effect Size', 'P-value', 'Degree', 'Group']

dfraw.to_csv(
    os.path.join(plot_dir, save_name + "_upsetplot.csv"),
    index=False
)

dfraw.iloc[40:70]

dfdata = prepare_upset_data(dfraw)
dfdata = dfdata.sort_values(by=['Group','Observed'], ascending=[True,False])

###CHATGPT OPTIMIZED FOR ANY NUMBER OF GROUPS
def kplot(df, size=(30,12)):
    """
    data : pd.DataFrame
    data is a dataframe with columns "Motifs" and "Counts"
    where "Motifs" is a list of lists e.g. [['PFC','LS'],['LS']]
    and "Counts" is a simple array of integers
    """
    motiflabels = df['Motifs'].to_list()
    data = up.from_memberships(motiflabels, data=df['Observed'].to_numpy())
    xlen = df.shape[0]
    xticks = np.arange(xlen)
    uplot = up.UpSet(data, sort_by=None)  # sort_by='cardinality'
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 3], 'height_ratios': [3, 1]})
    fig.set_size_inches(size)
    ax[1, 0].set_ylabel("Set Totals")
    uplot.plot_matrix(ax[1, 1])
    uplot.plot_totals(ax[1, 0])
    ax[0, 0].axis('off')
    ax[0, 1].spines['bottom'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    
    width = 0.35
    dodge = width / 2
    x = np.arange(8)
    ax[1, 0].set_title("Totals")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].set_xlim(ax[1, 1].get_xlim())
    
    # Get unique group labels and create color map for each group dynamically
    unique_groups = df['Group'].unique()
    colorlist = ['red', 'darkblue', 'black', 'green', 'purple']  # Extend as needed
    color_map = {group: colorlist[i % len(colorlist)] for i, group in enumerate(unique_groups)}
    
    # Map the colors based on the group
    cs = [color_map[group] for group in df['Group']]
    
    # Plot the bars with colors
    ax[0, 1].bar(xticks - dodge, df['Observed'].to_numpy(), width=width, label="Observed", align="center", color=cs, edgecolor='lightgray')
    ax[0, 1].bar(xticks + dodge, df['Expected'].to_numpy(), yerr=df['Expected SD'].to_numpy(), width=width / 2, label="Expected", align="center", color='gray', alpha=0.5, ecolor='lightgray')
    
    # Draw significance asterisks
    for group in unique_groups:
        group_ids = np.where(df['Group'].to_numpy() == group)[0]
        for i in group_ids:
            ax[0, 1].text(xticks[i] - 0.5 * dodge, df['Observed'].to_numpy()[i] + 1, "*", fontsize=12, color=color_map[group])

    # Hide axis ticks and grids
    ax[0, 1].xaxis.grid(False)
    ax[0, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.grid(False)
    
    fig.tight_layout()
    return fig, ax

fig, _ = kplot(dfdata)
fig.savefig(os.path.join(plot_dir, save_name + "_upsetplot_gpt.pdf"))

def kplot(df, size=(30,12)):
    """
    data : pd.DataFrame
    data is a dataframe with columns "Motifs" and "Counts"
    where "Motifs" is a list of lists e.g. [['PFC','LS'],['LS']]
    and "Counts" is a simple array of integers
    """
    motiflabels = df['Motifs'].to_list()
    data = up.from_memberships(motiflabels,data=df['Observed'].to_numpy())
    xlen = df.shape[0]
    xticks = np.arange(xlen)
    uplot = up.UpSet(data, sort_by=None) #sort_by='cardinality'
    fig,ax=plt.subplots(2,2,gridspec_kw={'width_ratios': [1, 3], 'height_ratios':[3,1]})
    fig.set_size_inches(size)
    ax[1,0].set_ylabel("Set Totals")
    uplot.plot_matrix(ax[1,1])
    uplot.plot_totals(ax[1,0])
    ax[0,0].axis('off')
    ax[0,1].spines['bottom'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    #ax[0,1].set_xticks([],[])
    width=0.35
    dodge=width/2
    x = np.arange(8)
    ax[1,0].set_title("Totals")
    ax[0,1].set_ylabel("Counts")
    #ax[0,1].set_xlim(-width,8)
    ax[0,1].set_xlim(ax[1,1].get_xlim())
    #ax[0,1].set_xticks(ax[1,1].get_xticks())
    ox = xticks-dodge
    ex = xticks+dodge
    #colorlist = ['cyan','darkgray','darkgray','red']
    colorlist = ['red','darkblue','black','black']
    cs = [colorlist[i-1] for i in df['Group']]
    ax[0,1].bar(ox,df['Observed'].to_numpy(),width=width,label="Observed", align="center",color=cs, edgecolor='lightgray')
    ax[0,1].bar(ex,df['Expected'].to_numpy(),yerr=df['Expected SD'].to_numpy(),width=width/2,label="Expected", align="center",color='gray',alpha=0.5,ecolor='lightgray')
    grp_ = dfdata['Group'].to_numpy()
    idsig = np.concatenate([np.where(grp_ == 1)[0],np.where(grp_ == 2)[0]])
    [ax[0,1].text(ox[idsig][i]-0.5*dodge,df['Observed'].to_numpy()[idsig][i]+1,s="*") for i in range(idsig.shape[0])]
    # 
    ax[0,1].xaxis.grid(False)
    ax[0,1].xaxis.set_visible(False)
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].xaxis.grid(False)
    #ax[0,1].legend()
    fig.tight_layout()
    return fig,ax

fig,_ = kplot(dfdata)

fig.savefig(os.path.join(plot_dir, save_name + "_upsetplot.pdf"))

df.astype(bool).sum()