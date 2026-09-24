"""
Cross-Target Nearest Neighbor Cluster Pairing

Across datasets of two clustered single-molecule localization microscopy (SMLM) targets, pair each cluster in the first set with its nearest-neighbor cluster in the second set when their centers fall within a user-defined radius, and export paired and unpaired clusters for both targets.

Author: Tanja Menche
Affiliation: Research group of Mike Heilemann, Goethe University Frankfurt am Main, Germany
Version: v1.0.0
Date: 2026-09-23

About this script:
------------------
- Finds the appropriate dbscan and centers files in each user-specified input folder, based on user-provided file endings.
- Computes colocalization between the two datasets with a user-defined radius by defining paired clusters as colocalized and unpaired clusters as non-colocalized.
- For each point in input1, finds the closest point in input2 within the user-defined radius (in pixels); if such a point exists, the pair is considered colocalized.
- Optionally applies an x-shift to the coordinates.
- Outputs HDF5 and YAML files for colocalized and non-colocalized ("out") clusters and centers.
- All output files are saved in a new folder called 'output_colocalization' inside each input folder, with filenames based on the input files and suffixed with '_colocalized' or '_out'.

Parts of this script were developed with assistance from OpenAI's ChatGPT and Cursor AI. The resulting code was reviewed, modified, and validated by the author.

User Inputs (edit the variables below):
---------------------------------------
- input1_folder: Path to the first input folder (data from first target)
- input2_folder: Path to the second input folder (data from second target)
- input1_dbscan_ending: File ending for the dbscan file in input1_folder (e.g., '_dbscan.hdf5')
- input1_centers_ending: File ending for the centers file in input1_folder (e.g., '_centers.hdf5')
- input2_dbscan_ending: File ending for the dbscan file in input2_folder (e.g., '_dbscan.hdf5')
- input2_centers_ending: File ending for the centers file in input2_folder (e.g., '_centers.hdf5')
- CENTERS_ONLY_MODE: True or False (see following section "Modes")
- radius: Colocalization radius in px (float)
- x_shift: Optional x-shift in px (float or None)

Modes
-----
1. Full mode (default): requires centers + dbscan files per input folder.
   Colocalization is determined from center pairs; dbscan localizations are split
   into colocalized / out by cluster group.

2. Centers-only mode: set CENTERS_ONLY_MODE = True, or set both dbscan endings to
   None. Only input1_centers_ending and input2_centers_ending are required.
   Colocalization is checked directly between center points; only colocalized and
   out centers HDF5/YAML files are written (no dbscan outputs).

Information about the input data:
--------------------------------
The script takes two input folders, one per imaging target (input1 and input2). In each folder it looks for one cluster-centers HDF5 file and one DBSCAN localizations HDF5 file, selected by the user-defined file endings.
The cluster-centers HDF5 files contain one row per cluster in the 'locs' dataset. Pairing uses the cluster coordinates ('x_mean'/'y_mean', or 'x'/'y' if those are not present). The cluster identity is taken from 'group_mean' or 'group'. Coordinates are assumed to be in pixels, and the pairing radius is applied in pixels.
The DBSCAN HDF5 files contain the member localizations of those clusters in the 'locs' dataset, with a 'group' field that links each localization to its cluster. After pairing, localizations are split into colocalized and unpaired sets by this group ID.
Each HDF5 file is accompanied by a YAML file with the same filename ('.hdf5' replaced by '.yaml')for compatibility with the Picasso Software. Corresponding HDF5 files and YAML files can be generated with the Picasso Software (https://github.com/jungmannlab/picasso) version v0.7.3 from SMLM data (find more information in the README.md file).
Cluster-centers HDF5 files are identified by the "input1_centers_ending" / "input2_centers_ending" and DBSCAN HDF5 files are identified by the "input1_dbscan_ending" / "input2_dbscan_ending". HDF5 files from input1_folder and input2_folder are matched based on their base name, which is extracted by substracting the user given endings from the file name.

How to use:
-----------
1. Edit the variables in the USER INPUT SECTION below to match your data and analysis requirements.
2. Run the script.
3. For both input folders, the script saves HDF5 files of paired (colocalized) and unpaired (out) clusters and their centers, each accompanied by a YAML file of the same filename for compatibility with the Picasso Software. These files are written to a new folder named output_colocalization inside the corresponding input folder, with filenames based on the input files and suffixed with _colocalized or _out.
"""

import h5py
import numpy as np
import os
import yaml
import glob

# ====== USER INPUT SECTION ======
input1_folder = r"C:\cross-target_NN_cluster_pairing\example_data\input_data\protein1"
input2_folder = r"C:\cross-target_NN_cluster_pairing\example_data\input_data\protein2"
input1_dbscan_ending = "_dbscan.hdf5"
input1_centers_ending = "_dbscan_centers.hdf5"
input2_dbscan_ending = "_dbscan.hdf5"
input2_centers_ending = "_dbscan_centers.hdf5"

# Set True, or set both dbscan endings to None, to run without dbscan files.
CENTERS_ONLY_MODE = False

radius = 0.4  # User-defined radius in px
x_shift = None  # User-defined shift in x direction in px (set to None if no shift is needed)
# ===== END USER INPUT SECTION =====


def read_hdf5_data(file_path, dataset_name):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:]
        return data
    except OSError as e:
        print(f"Error reading file {file_path}: {e}")
        raise


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_colocalized_indices(x1, y1, x2, y2, radius):
    matched_indices = []
    for i, (x1_val, y1_val) in enumerate(zip(x1, y1)):
        distances = euclidean_distance(x1_val, y1_val, x2, y2)
        within_radius_indices = np.where(distances <= radius)[0]
        if len(within_radius_indices) > 0:
            min_distance_index = within_radius_indices[np.argmin(distances[within_radius_indices])]
            matched_indices.append((i, min_distance_index))
    return matched_indices


def save_hdf5_data(file_path, dataset_name, data):
    try:
        print(f"Saving data to {file_path} under dataset {dataset_name}")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(dataset_name, data=data)
        print(f"Data saved successfully to {file_path}")
    except OSError as e:
        print(f"Error writing to file {file_path}: {e}")
        raise


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return list(yaml.safe_load_all(f))


def read_yaml_optional(centers_path):
    """Read companion YAML if present; return None otherwise."""
    yaml_path = centers_path.replace('.hdf5', '.yaml')
    if os.path.exists(yaml_path):
        return read_yaml(yaml_path)
    return None


def write_yaml_optional(file_path, data):
    if data is not None:
        write_yaml(file_path, data)


def use_centers_only_mode():
    """True when only centers files should be processed."""
    if CENTERS_ONLY_MODE:
        return True
    return input1_dbscan_ending is None and input2_dbscan_ending is None


def write_yaml(file_path, data):
    with open(file_path, 'w') as f:
        yaml.safe_dump_all(data, f)
    print(f"YAML file saved to {file_path}")


def apply_x_shift(properties, dbscan, x_shift):
    print(f"Applying x_shift of {x_shift}")
    if 'x_mean' in properties.dtype.names:
        properties['x_mean'] += x_shift
    if 'x' in properties.dtype.names:
        properties['x'] += x_shift
    if dbscan is not None and 'x' in dbscan.dtype.names:
        dbscan['x'] += x_shift
    return properties, dbscan


def apply_x_shift_centers(properties, x_shift):
    """Apply x-shift to centers/locs only (centers-only mode)."""
    print(f"Applying x_shift of {x_shift} to centers")
    properties = np.copy(properties)
    if 'x_mean' in properties.dtype.names:
        properties['x_mean'] += x_shift
    if 'x' in properties.dtype.names:
        properties['x'] += x_shift
    return properties


def extract_coordinates(properties):
    print("Extracting coordinates")
    if 'x_mean' in properties.dtype.names and 'y_mean' in properties.dtype.names:
        print("Using x_mean and y_mean")
        return properties['x_mean'], properties['y_mean']
    elif 'x' in properties.dtype.names and 'y' in properties.dtype.names:
        print("Using x and y")
        return properties['x'], properties['y']
    else:
        raise ValueError("Properties file does not contain the required coordinates ('x_mean', 'y_mean' or 'x', 'y').")


def extract_group_data(properties):
    print("Extracting group data")
    if 'group_mean' in properties.dtype.names:
        print("Using group_mean")
        return properties['group_mean']
    elif 'group' in properties.dtype.names:
        print("Using group")
        return properties['group']
    else:
        raise ValueError("Properties file does not contain the required group data ('group_mean' or 'group').")


def find_file_with_ending(folder, ending):
    pattern = os.path.join(folder, f"*{ending}")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file ending with '{ending}' found in folder '{folder}'")
    if len(files) > 1:
        print(f"Warning: Multiple files ending with '{ending}' found in folder '{folder}'. Using the first one: {files[0]}")
    return files[0]

def find_all_files_with_ending(folder, ending):
    pattern = os.path.join(folder, f"*{ending}")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No files ending with '{ending}' found in folder '{folder}'"
        )

    return sorted(files)

def get_base_name(file_path, ending):
    """
    Removes the user-defined ending from the filename.

    Example:
    sample1_centers_filter_simulation.hdf5
    -> sample1
    """
    filename = os.path.basename(file_path)

    if not filename.endswith(ending):
        raise ValueError(
            f"File '{filename}' does not end with '{ending}'"
        )

    return filename[:-len(ending)]

def make_output_paths(input1_dbscan, input1_centers, input2_dbscan, input2_centers):
    def get_out_folder_and_base(input_file):
        folder = os.path.dirname(input_file)
        out_folder = os.path.join(folder, "output_colocalization")
        os.makedirs(out_folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_file))[0]
        return out_folder, base

    out1_folder, base1_centers = get_out_folder_and_base(input1_centers)
    out1_folder, base1_dbscan = get_out_folder_and_base(input1_dbscan)
    out2_folder, base2_centers = get_out_folder_and_base(input2_centers)
    out2_folder, base2_dbscan = get_out_folder_and_base(input2_dbscan)

    output_paths = {
        'input1_centers_colocalized': os.path.join(out1_folder, f"{base1_centers}_colocalized.hdf5"),
        'input2_centers_colocalized': os.path.join(out2_folder, f"{base2_centers}_colocalized.hdf5"),
        'input1_centers_out': os.path.join(out1_folder, f"{base1_centers}_out.hdf5"),
        'input2_centers_out': os.path.join(out2_folder, f"{base2_centers}_out.hdf5"),
        'input1_dbscan_colocalized': os.path.join(out1_folder, f"{base1_dbscan}_colocalized.hdf5"),
        'input2_dbscan_colocalized': os.path.join(out2_folder, f"{base2_dbscan}_colocalized.hdf5"),
        'input1_dbscan_out': os.path.join(out1_folder, f"{base1_dbscan}_out.hdf5"),
        'input2_dbscan_out': os.path.join(out2_folder, f"{base2_dbscan}_out.hdf5"),
    }
    return output_paths


def process_files(input1_dbscan, input1_centers, input2_dbscan, input2_centers, radius, output_paths,
                  x_shift=None):
    print("Starting process_files")

    # Verify files exist
    for file_path in [input1_dbscan, input1_centers, input2_dbscan, input2_centers]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # Read the properties and dbscan information
    properties1 = read_hdf5_data(input1_centers, 'locs')
    properties2 = read_hdf5_data(input2_centers, 'locs')
    dbscan1 = read_hdf5_data(input1_dbscan, 'locs')
    dbscan2 = read_hdf5_data(input2_dbscan, 'locs')

    # Apply x shift if specified
    if x_shift is not None:
        properties1, dbscan1 = apply_x_shift(properties1, dbscan1, x_shift)

    # Read the YAML metadata if available
    input1_yaml = read_yaml(input1_centers.replace('.hdf5', '.yaml'))
    input2_yaml = read_yaml(input2_centers.replace('.hdf5', '.yaml'))

    # Extract coordinates based on available keys
    x1, y1 = extract_coordinates(properties1)
    x2, y2 = extract_coordinates(properties2)

    # Extract group data based on available keys
    group_data1 = extract_group_data(properties1)
    group_data2 = extract_group_data(properties2)

    # Find colocalized indices
    matched_indices = find_colocalized_indices(x1, y1, x2, y2, radius)

    print(f"Matched Indices: {matched_indices}")

    # Extract unique group values from properties
    matched_indices_set1 = set()
    matched_indices_set2 = set()

    for i, j in matched_indices:
        matched_indices_set1.add(i)
        matched_indices_set2.add(j)

    colocalized_group_means1 = np.unique(group_data1[list(matched_indices_set1)])
    colocalized_group_means2 = np.unique(group_data2[list(matched_indices_set2)])

    # Filter dbscan based on colocalized group values
    colocalized_dbscan1 = []
    colocalized_dbscan2 = []
    out_dbscan1 = []
    out_dbscan2 = []

    for group_mean in colocalized_group_means1:
        colocalized_dbscan1.extend(dbscan1[dbscan1['group'] == group_mean])

    for group_mean in colocalized_group_means2:
        colocalized_dbscan2.extend(dbscan2[dbscan2['group'] == group_mean])

    for group_mean in np.unique(dbscan1['group']):
        if group_mean not in colocalized_group_means1:
            out_dbscan1.extend(dbscan1[dbscan1['group'] == group_mean])

    for group_mean in np.unique(dbscan2['group']):
        if group_mean not in colocalized_group_means2:
            out_dbscan2.extend(dbscan2[dbscan2['group'] == group_mean])

    # Save colocalized and out data
    save_hdf5_data(output_paths['input1_centers_colocalized'], 'locs', np.array(properties1[list(matched_indices_set1)]))
    save_hdf5_data(output_paths['input2_centers_colocalized'], 'locs', np.array(properties2[list(matched_indices_set2)]))
    save_hdf5_data(output_paths['input1_centers_out'], 'locs',
                   np.array(properties1[[i for i in range(len(properties1)) if i not in matched_indices_set1]]))
    save_hdf5_data(output_paths['input2_centers_out'], 'locs',
                   np.array(properties2[[j for j in range(len(properties2)) if j not in matched_indices_set2]]))

    # Save dbscan data
    save_hdf5_data(output_paths['input1_dbscan_colocalized'], 'locs', np.array(colocalized_dbscan1))
    save_hdf5_data(output_paths['input2_dbscan_colocalized'], 'locs', np.array(colocalized_dbscan2))
    save_hdf5_data(output_paths['input1_dbscan_out'], 'locs', np.array(out_dbscan1))
    save_hdf5_data(output_paths['input2_dbscan_out'], 'locs', np.array(out_dbscan2))

    # Save corresponding YAML files for output
    write_yaml(output_paths['input1_centers_colocalized'].replace('.hdf5', '.yaml'), input1_yaml)
    write_yaml(output_paths['input2_centers_colocalized'].replace('.hdf5', '.yaml'), input2_yaml)
    write_yaml(output_paths['input1_centers_out'].replace('.hdf5', '.yaml'), input1_yaml)
    write_yaml(output_paths['input2_centers_out'].replace('.hdf5', '.yaml'), input2_yaml)

    write_yaml(output_paths['input1_dbscan_colocalized'].replace('.hdf5', '.yaml'), input1_yaml)
    write_yaml(output_paths['input2_dbscan_colocalized'].replace('.hdf5', '.yaml'), input2_yaml)
    write_yaml(output_paths['input1_dbscan_out'].replace('.hdf5', '.yaml'), input1_yaml)
    write_yaml(output_paths['input2_dbscan_out'].replace('.hdf5', '.yaml'), input2_yaml)

    print("Process completed successfully")

def make_output_paths_centers_only(input1_centers, input2_centers):
    def get_out_folder_and_base(input_file):
        folder = os.path.dirname(input_file)
        out_folder = os.path.join(folder, "output_colocalization")
        os.makedirs(out_folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_file))[0]
        return out_folder, base

    out1_folder, base1 = get_out_folder_and_base(input1_centers)
    out2_folder, base2 = get_out_folder_and_base(input2_centers)

    output_paths = {
        'input1_centers_colocalized': os.path.join(
            out1_folder, f"{base1}_colocalized.hdf5"
        ),
        'input2_centers_colocalized': os.path.join(
            out2_folder, f"{base2}_colocalized.hdf5"
        ),
        'input1_centers_out': os.path.join(
            out1_folder, f"{base1}_out.hdf5"
        ),
        'input2_centers_out': os.path.join(
            out2_folder, f"{base2}_out.hdf5"
        ),
    }

    return output_paths

def process_files_centers_only(
    input1_centers,
    input2_centers,
    radius,
    output_paths,
    x_shift=None
):
    print("Starting centers-only processing")

    # Verify files exist
    for file_path in [input1_centers, input2_centers]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # Read center data
    properties1 = read_hdf5_data(input1_centers, 'locs')
    properties2 = read_hdf5_data(input2_centers, 'locs')

     # Apply optional x shift
    if x_shift is not None:
        properties1 = apply_x_shift_centers(properties1, x_shift)

    # Read YAML metadata if available
    input1_yaml = read_yaml_optional(input1_centers)
    input2_yaml = read_yaml_optional(input2_centers)

    # Extract coordinates
    x1, y1 = extract_coordinates(properties1)
    x2, y2 = extract_coordinates(properties2)

    # Find colocalized indices
    matched_indices = find_colocalized_indices(x1, y1, x2, y2, radius)

    matched_indices_set1 = set()
    matched_indices_set2 = set()

    for i, j in matched_indices:
        matched_indices_set1.add(i)
        matched_indices_set2.add(j)

         # Create output arrays
    coloc1 = np.array(properties1[list(matched_indices_set1)])
    coloc2 = np.array(properties2[list(matched_indices_set2)])

    out1 = np.array([
        properties1[i]
        for i in range(len(properties1))
        if i not in matched_indices_set1
    ])

    out2 = np.array([
        properties2[j]
        for j in range(len(properties2))
        if j not in matched_indices_set2
    ])

     # Save HDF5 outputs
    save_hdf5_data(
        output_paths['input1_centers_colocalized'],
        'locs',
        coloc1
    )

    save_hdf5_data(
        output_paths['input2_centers_colocalized'],
        'locs',
        coloc2
    )

    save_hdf5_data(
        output_paths['input1_centers_out'],
        'locs',
        out1
    )

    save_hdf5_data(
        output_paths['input2_centers_out'],
        'locs',
        out2
    )

    # Save YAML outputs if metadata exists
    write_yaml_optional(
        output_paths['input1_centers_colocalized'].replace('.hdf5', '.yaml'),
        input1_yaml
    )

    write_yaml_optional(
        output_paths['input2_centers_colocalized'].replace('.hdf5', '.yaml'),
        input2_yaml
    )

    write_yaml_optional(
        output_paths['input1_centers_out'].replace('.hdf5', '.yaml'),
        input1_yaml
    )

    write_yaml_optional(
        output_paths['input2_centers_out'].replace('.hdf5', '.yaml'),
        input2_yaml
    )

    print("Centers-only processing completed successfully")



if __name__ == "__main__":

    # ==========================================
    # CENTERS-ONLY MODE
    # ==========================================
    if use_centers_only_mode():

        if input1_dbscan_ending or input2_dbscan_ending:
            print(
                "Centers-only mode: dbscan endings are ignored; "
                "only centers files are used."
            )

        # Find all center files
        input1_center_files = find_all_files_with_ending(
            input1_folder,
            input1_centers_ending
        )

        input2_center_files = find_all_files_with_ending(
            input2_folder,
            input2_centers_ending
        )

        # Build dictionary using base names
        input2_dict = {
            get_base_name(f, input2_centers_ending): f
            for f in input2_center_files
        }

        # Process matching pairs
        for input1_file in input1_center_files:
            base_name = get_base_name(
                input1_file,
                input1_centers_ending
            )

            if base_name not in input2_dict:
                print(
                    f"WARNING: No matching file found in input2 for '{base_name}'"
                )
                continue

            input2_file = input2_dict[base_name]

            print("\n====================================")
            print(f"Processing pair: {base_name}")
            print("====================================")

            output_paths = make_output_paths_centers_only(
                input1_file,
                input2_file
            )

            process_files_centers_only(
                input1_file,
                input2_file,
                radius,
                output_paths,
                x_shift
            )

    # ==========================================
    # FULL MODE
    # ==========================================
    else:

        if input1_dbscan_ending is None or input2_dbscan_ending is None:
            raise ValueError(
                "Both input1_dbscan_ending and input2_dbscan_ending are required "
                "in full mode."
            )

        # Find all files
        input1_center_files = find_all_files_with_ending(
            input1_folder,
            input1_centers_ending
        )

        input2_center_files = find_all_files_with_ending(
            input2_folder,
            input2_centers_ending
        )

        input1_dbscan_files = find_all_files_with_ending(
            input1_folder,
            input1_dbscan_ending
        )

        input2_dbscan_files = find_all_files_with_ending(
            input2_folder,
            input2_dbscan_ending
        )

        # Create dictionaries
        input1_centers_dict = {
            get_base_name(f, input1_centers_ending): f
            for f in input1_center_files
        }

        input2_centers_dict = {
            get_base_name(f, input2_centers_ending): f
            for f in input2_center_files
        }

        input1_dbscan_dict = {
            get_base_name(f, input1_dbscan_ending): f
            for f in input1_dbscan_files
        }

        input2_dbscan_dict = {
            get_base_name(f, input2_dbscan_ending): f
            for f in input2_dbscan_files
        }

        # Use intersection of all sets
        common_base_names = (
            set(input1_centers_dict.keys())
            & set(input2_centers_dict.keys())
            & set(input1_dbscan_dict.keys())
            & set(input2_dbscan_dict.keys())
        )

        if not common_base_names:
            raise ValueError("No matching file sets found.")

        for base_name in sorted(common_base_names):

            print("\n====================================")
            print(f"Processing pair: {base_name}")
            print("====================================")

            input1_centers = input1_centers_dict[base_name]
            input2_centers = input2_centers_dict[base_name]
            input1_dbscan = input1_dbscan_dict[base_name]
            input2_dbscan = input2_dbscan_dict[base_name]

            output_paths = make_output_paths(
                input1_dbscan,
                input1_centers,
                input2_dbscan,
                input2_centers
            )

            process_files(
                input1_dbscan,
                input1_centers,
                input2_dbscan,
                input2_centers,
                radius,
                output_paths,
                x_shift,
            )