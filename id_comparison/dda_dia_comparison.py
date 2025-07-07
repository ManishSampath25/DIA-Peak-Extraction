import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pyteomics import mgf
from pyteomics import mzml
from sklearn import metrics
import re

### paths & params ###
DDA_PATH = 'D:Manish/projects/umpire_data/maestro/HEK/dda/MAESTRO-HEK_DDA-identified_variants_merged_protein_regions-main.tsv'
DIA_PATH = 'D:Manish/projects/umpire_data/maestro/HEK/dia/2p5ms/MAESTRO-HEK_DIA_No_Clustering-identified_variants_merged_protein_regions-main.tsv'
O_PATH = 'D:Manish/projects/umpire_data/'
DDA_DIR = 'E:/Datasets/MSV000095975_PXD046453-Astral/HEK_DDA/'
DDA_FILE_TYPE = 'mzml'
DIA_DIR = 'E:/Datasets/MSV000095975_PXD046453-Astral/HEK_DIA-Umpire/'
DIA_FILE_TYPE = 'mgf'
SPLIT_REP_GROUPS = ''


### standardize mods ###
def standardize(row):
    seq = row['Unmodified_sequence']
    mass_offset = row['Sum_of_peptide_mass_offsets']
    return f'{seq}{mass_offset}'

### venn ###
def plot_venn2(dda_peps, dia_peps):
    plt.figure(figsize=(6, 6))
    venn2(
        [set(dda_peps), set(dia_peps)],
        set_labels=('DDA', 'DIA'),
    )
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/Venn.png')
    plt.close()

def plot_bar(dda_seqs, dia_seqs, dia):
    dda_seqs = set(dda_seqs)
    dia_seqs_q1 = set(dia_seqs[dia['DisplayRep_Filename'].str.contains('Q1')])
    dia_seqs_q2 = set(dia_seqs[dia['DisplayRep_Filename'].str.contains('Q2')])
    dia_seqs_q3 = set(dia_seqs[dia['DisplayRep_Filename'].str.contains('Q3')])

    q1_nums = len(dda_seqs.intersection(dia_seqs_q1))
    q2_nums = len(dda_seqs.intersection(dia_seqs_q2))
    q3_nums = len(dda_seqs.intersection(dia_seqs_q3))

    quality_scores = ['Q1', 'Q2', 'Q3']
    num_peptides = [q1_nums, q2_nums, q3_nums]

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.bar(quality_scores, num_peptides)
    plt.xlabel('Quality Score')
    plt.ylabel('Number of Peptide Variants')
    plt.title('Peptide Variants by Quality Tier')
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/bar.png')
    plt.close()

# This assumes that there is no duplicates in either dda or dia, which shouldn't be a problem since we are working at
# the variant level
def shared_spectra(dda, dia, dda_seqs, dia_seqs, int_peps):
    # Drop duplicates: keep first occurrence for each peptide
    dda_map = dda.copy()
    dda_map['peptide'] = dda_seqs
    dda_filtered = dda_map[dda_map['peptide'].isin(int_peps)].drop_duplicates('peptide').set_index('peptide')

    dia_map = dia.copy()
    dia_map['peptide'] = dia_seqs
    dia_filtered = dia_map[dia_map['peptide'].isin(int_peps)].drop_duplicates('peptide').set_index('peptide')

    # Align to int_peps
    dda_filtered = dda_filtered.reindex(int_peps)
    dia_filtered = dia_filtered.reindex(int_peps)

    cols = ['Representative_Filename', 'Representative_Scan', 'Charge']

    combined = pd.concat(
        [dda_filtered[cols], dia_filtered[cols]],
        axis=1,
        keys=['dda', 'dia']
    )

    return combined

def cosine(row):
    dda_spec = row['dda', 'spectrum']
    dia_spec = row['dia', 'spectrum']
    vec_size = max(set(dda_spec.keys()).union(dia_spec.keys()))
    dda_vec = np.zeros(int(vec_size+2))
    for key, val in dda_spec.items():
        dda_vec[int(key)] = val
    dia_vec = np.zeros(int(vec_size+2))
    for key, val in dia_spec.items():
        dia_vec[int(key)] = val
    return np.dot(dda_vec, dia_vec) / (np.linalg.norm(dda_vec) * np.linalg.norm(dia_vec))


def retrieve_spectrum(file, target, filetype):
    if filetype == 'mgf':
        with mgf.read(file) as f:
            for spectrum in f:
                title = spectrum.get('params', {}).get('title', '')
                if target in title:
                    return dict(zip(spectrum['m/z array'], spectrum['intensity array']))
    elif filetype == 'mzml':
        with mzml.MzML(file) as f:
            for spectrum in f:
                spec_id = spectrum.get('id', '')
                if target in spec_id:
                    return dict(zip(spectrum['m/z array'], spectrum['intensity array']))
    return None


def retrieve_spectra(result, file, filetype):
    if filetype == 'mgf':
        with mgf.read(file) as f:
            for spectrum in f:
                title = spectrum.get('params', {}).get('title', '')
                match = re.search(r'Q2\.(\d+)', title)
                if match:
                    scan_number = match.group(1)
                    if scan_number in result:
                        result[scan_number] = dict(zip(spectrum['m/z array'], spectrum['intensity array']))

    elif filetype == 'mzml':
        with mzml.MzML(file) as f:
            for spectrum in f:
                spec_id = spectrum.get('id', '')
                match = re.search(r'scan=(\d+)', spec_id)
                if match:
                    scan_number = match.group(1)
                    if scan_number in result:
                        result[scan_number] = dict(zip(spectrum['m/z array'], spectrum['intensity array']))

    return result


def compute_similarity(overlap_df):
    dda_files = set(overlap_df['dda', 'Representative_Filename'])
    dia_files = set(overlap_df['dia', 'Representative_Filename'])

    # extract spectra
    dda_dict = {str(key) : None for key in overlap_df['dda', 'Representative_Scan']}
    for dda_file in dda_files:
        dda_file = os.path.join(DDA_DIR, os.path.basename(dda_file))
        dda_dict = retrieve_spectra(dda_dict, dda_file, DDA_FILE_TYPE)
        print(f'Processed {dda_file}')
    overlap_df[('dda', 'spectrum')] = overlap_df[('dda', 'Representative_Scan')].astype(str).map(dda_dict)

    dia_dict = {str(key) : None for key in overlap_df['dia', 'Representative_Scan']}
    for dia_file in dia_files:
        dia_file = os.path.join(DIA_DIR, os.path.basename(dia_file))
        dia_dict = retrieve_spectra(dia_dict, dia_file, DIA_FILE_TYPE)
        print(f'Processed {dia_file}')

    overlap_df[('dia', 'spectrum')] = overlap_df[('dia', 'Representative_Scan')].astype(str).map(dia_dict)

    # filter None spectrums + match charges
    overlap_df = overlap_df.dropna()
    charge_match = overlap_df[('dda', 'Charge')] == overlap_df[('dia', 'Charge')]
    overlap_df = overlap_df[charge_match]

    scores = dict()
    for quality in ['Q1', 'Q2', 'Q3']:
        q_filt_df = overlap_df[overlap_df['dia', 'Representative_Filename'].str.contains(quality)]
        scores[quality] = q_filt_df.apply(cosine, axis=1)

    return scores

def main():

    # read in data
    dda = pd.read_csv(DDA_PATH, sep='\t')
    dia = pd.read_csv(DIA_PATH, sep='\t')

    # process sequences
    dda_seqs = dda.apply(standardize, axis=1)
    dia_seqs = dia.apply(standardize, axis=1)

    # plot venn diagram
    plot_venn2(dda_seqs, dia_seqs)
    plot_bar(dda_seqs, dia_seqs, dia)

    # intersect and find spectra ids
    int_peps = dia_seqs[dia_seqs.isin(dda_seqs)].reset_index(drop=True)
    overlap_df = shared_spectra(dda, dia, dda_seqs, dia_seqs, int_peps)

    # extract spectra and compute similarity
    scores = compute_similarity(overlap_df)

    # plot
    data = [scores['Q1'], scores['Q2'], scores['Q3']]
    labels = ['Q1', 'Q2', 'Q3']
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)
    plt.title('Cosine Similarity by Quality Tier')
    plt.ylabel('Cosine Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/HEK_boxplot.png')

    return 0


if __name__ == "__main__":
    main()