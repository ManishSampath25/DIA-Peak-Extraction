import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pyteomics import mzml, mgf
import re

# paths and params
DDA_PATH = ''
DIA_PATH = ''
DDA_PSM = ''
DIA_PSM = ''
DIA_PEPXML_DIR = ['']
DDA_PEPXML_DIR = ['']
O_PATH = ''
DDA_FILE_TYPE


# Preprocess
def extract_sequences(row):
    if row['Modified Sequence']:
        return row['Modified Sequence']
    return row['Peptide Sequence']

# plot venn diagram
def plot_venn2(dda_peps, dia_peps):
    plt.figure(figsize=(6, 6))
    venn2(
        [set(dda_peps), set(dia_peps)],
        set_labels=('DDA', 'DIA'),
    )
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/Venn.png')
    plt.close()


def shared_spectra(dda, dia, dda_seqs, dia_seqs, int_peps):
    dda_map = dda.copy()
    dda_map['peptide'] = dda_seqs
    dda_filtered = dda_map[dda_map['peptide'].isin(int_peps)].drop_duplicates('peptide').set_index('peptide')

    dia_map = dia.copy()
    dia_map['peptide'] = dia_seqs
    dia_filtered = dia_map[dia_map['peptide'].isin(int_peps)].drop_duplicates('peptide').set_index('peptide')

    dda_filtered = dda_filtered.reindex(int_peps)
    dia_filtered = dia_filtered.reindex(int_peps)

    cols = ['Spectrum File', 'Spectrum', 'Charge']

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
    dda_files = set(overlap_df['dda', 'Spectrum File'])
    dia_files = set(overlap_df['dia', 'Spectrum File'])

    # extract spectra
    dda_dict = {re.search(r'Q2\.(\d+)', title).group(1): None for title in overlap_df['dda', 'Spectrum']}
    for dda_file in dda_files:
        dda_file = os.path.join(DDA_PEPXML_DIR, os.path.basename(dda_file))
        dda_dict = retrieve_spectra(dda_dict, dda_file, DDA_FILE_TYPE)
        print(f'Processed {dda_file}')
    overlap_df[('dda', 'spectrum_array')] = overlap_df[('dda', 'Spectrum')].astype(str).map(dda_dict)

    dia_dict = {re.search(r'Q2\.(\d+)', key).group(1) : None for key in overlap_df['dia', 'Spectrum']}
    for dia_file in dia_files:
        dia_file = os.path.join(DIA_PEPXML_DIR, os.path.basename(dia_file))
        dia_dict = retrieve_spectra(dia_dict, dia_file, DIA_FILE_TYPE)
        print(f'Processed {dia_file}')

    overlap_df[('dia', 'spectrum_array')] = overlap_df[('dia', 'Spectrum')].astype(str).map(dia_dict)

    # filter None spectrums + match charges
    overlap_df = overlap_df.dropna()
    charge_match = overlap_df[('dda', 'Charge')] == overlap_df[('dia', 'Charge')]
    overlap_df = overlap_df[charge_match]

    scores = dict()
    for quality in ['Q1', 'Q2', 'Q3']:
        q_filt_df = overlap_df[overlap_df['dia', 'Spectrum File'].str.contains(quality)]
        scores[quality] = q_filt_df.apply(cosine, axis=1)

    return scores



def main():

    dda = pd.read_csv(DDA_PATH, sep = '\t')
    dia = pd.read_csv(DIA_PATH, sep = '\t')

    dda_seqs = dda.apply(extract_sequences, axis=1)
    dia_seqs = dia.apply(extract_sequences, axis=1)

    plot_venn2(dda_seqs, dia_seqs)

    int_peps = dia_seqs[dia_seqs.isin(dda_seqs)].reset_index(drop=True)
    dda_psm = pd.read_csv(DDA_PSM, sep = '\t')
    dia_psm = pd.read_csv(DIA_PSM, sep = '\t')
    overlap_df = shared_spectra(dda_psm, dia_psm, dda_seqs, dia_seqs, int_peps)

    scores = compute_similarity(overlap_df)



