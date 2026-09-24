# Cross-Target Nearest Neighbor Cluster Pairing

Across datasets of two clustered single-molecule localization microscopy (SMLM) targets, pair each cluster in the first set with its nearest-neighbor cluster in the second set when their centers fall within a 
user-defined radius, and export paired and unpaired clusters for both targets.


## Overview

Finds the appropriate dbscan and centers files in each user-specified input folder, based on user-provided file endings.
- Computes colocalization between the two datasets with a user-defined radius by defining paired clusters as colocalized and unpaired clusters as non-colocalized.
- For each point in input1, finds the closest point in input2 within the user-defined radius (in pixels); if such a point exists, the pair is considered colocalized.
- Optionally applies an x-shift to the coordinates.
- Outputs HDF5 and YAML files for colocalized and non-colocalized ("out") clusters and centers.
- All output files are saved in a new folder called 'output_colocalization' inside each input folder, with filenames based on the input files and suffixed with '_colocalized' or '_out'.

Note: The colocalization matching is performed from input file 1 to input file 2, so the same center point in input file 2 can be matched to multiple points in input file 1. 
Therefore, the numbers of colocalized centers reported for the two input files can differ slightly.

Parts of this script were developed with assistance from OpenAI's ChatGPT and Cursor AI. The resulting code was reviewed, modified, and validated by the author.


## Requirements

* Python **3.11**
* PyYAML **6.0.1**
* h5py **3.9.0**
* numpy **1.26.4**

## Input Data

The script takes two input folders, one per imaging target (input1 and input2). In each folder it looks for one cluster-centers HDF5 file and one DBSCAN localizations HDF5 file 
(only if 'CENTERS_ONLY_MODE = False', see section **Modes**), selected by the user-defined file endings. The script was tested on HDF5 files with DBSCAN localization files and cluster center files 
generated from localization data via the clustering algorithm DBSCAN with the [Picasso Software](https://github.com/jungmannlab/picasso) version 7.3 and with the 
[PicassoBatchProcess](https://github.com/HeilemannLab/PicassoBatchProcess) Software.  
The cluster-centers HDF5 files contain one row per cluster in the 'locs' dataset. Pairing uses the cluster coordinates ('x_mean'/'y_mean', or 'x'/'y' if those are not present). 
The cluster identity is taken from 'group_mean' or 'group'. Coordinates are assumed to be in pixels, and the pairing radius is applied in pixels.
The DBSCAN HDF5 files contain the member localizations of those clusters in the 'locs' dataset, with a 'group' field that links each localization to its cluster. After pairing, 
localizations are split into colocalized and unpaired sets by this group ID.
Each HDF5 file is accompanied by a YAML file with the same filename ('.hdf5' replaced by '.yaml')for compatibility with the Picasso Software. Corresponding HDF5 files and YAML files 
can be generated with the Picasso Software version v0.7.3 from SMLM data (find more information in the README.md file).
Cluster-centers HDF5 files are identified by the "input1_centers_ending" / "input2_centers_ending" and DBSCAN HDF5 files are identified by the "input1_dbscan_ending" / 
"input2_dbscan_ending". HDF5 files from input1_folder and input2_folder are matched based on their base name, which is extracted by substracting the user given endings from the file name.

```
input_data/ 

├── protein1/ 

│ ├── cell1_ROI_dbscan.yaml

│ ├── cell1_ROI_dbscan.hdf5

│ ├── cell1_ROI_dbscan_centers.yaml

│ └── cell1_ROI_dbscan_centers.hdf5 

│ 
└── protein2/ 

  ├── cell1_ROI_dbscan.yaml
  
  ├── cell1_ROI_dbscan.hdf5
  
  ├── cell1_ROI_dbscan_centers.yaml
  
  └── cell1_ROI_dbscan_centers.hdf5   
```
  

## Configuration

Before running the script, edit the following variables in the **USER INPUT SECTION** section in `cross_target_NN_cluster_pairing.py`:
```python
input1_folder = r"C:\cross-target_NN_cluster_pairing\example_data\input_data\protein1"
input2_folder = r"C:\cross-target_NN_cluster_pairing\example_data\input_data\protein2"
input1_dbscan_ending = "_dbscan.hdf5"
input1_centers_ending = "_centers.hdf5"
input2_dbscan_ending = "_dbscan.hdf5"
input2_centers_ending = "_centers.hdf5"
CENTERS_ONLY_MODE: True or False (see following section "Modes")
radius = 0.4  # User-defined radius in px
x_shift = None  # User-defined shift in x direction in px (set to None if no shift is needed)
```

### Modes
1. Full mode (default): requires centers + dbscan files per input folder.
   Colocalization is determined from center pairs; dbscan localizations are split
   into colocalized / out by cluster group.

2. Centers-only mode: set CENTERS_ONLY_MODE = True, or set both dbscan endings to
   None. Only input1_centers_ending and input2_centers_ending are required.
   Colocalization is checked directly between center points; only colocalized and
   out centers HDF5/YAML files are written (no dbscan outputs).
   

## Installation
```PowerShell
conda create --name cross_target_NN_cluster_pairing python=3.11
conda activate cross_target_NN_cluster_pairing
cd filepath\cross_target_NN_cluster_pairing
conda install --file requirements.txt
```

## Usage

1. Place the YAML and HDF5 input files in the specified folders.
2. Open `cross_target_NN_cluster_pairing.py`.
3. Set variables in CONFIGURATION section.
4. Open environment:
```PowerShell
conda activate cross_target_NN_cluster_pairing
```
5. Navigate to the file path, where nearest_neighbor_distances.py is stored:
```PowerShell
cd filepath\cross_target_NN_cluster_pairing
```
6. Run:
```PowerShell
python cross_target_NN_cluster_pairing.py
```

## Output

For both input folders, the script saves HDF5 files of paired (colocalized) and unpaired (out) clusters and their centers, each accompanied by a YAML file of the same filename for 
compatibility with the Picasso Software. These files are written to a new folder named output_colocalization inside the corresponding input folder, with filenames based on the input files and 
suffixed with _colocalized or _out.


