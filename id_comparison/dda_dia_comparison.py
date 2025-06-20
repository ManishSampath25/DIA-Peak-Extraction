import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pyteomics import mgf

### paths & params ###
DDA_PATH = ''
DIA_PATH = ''
O_PATH = ''
MGF_DIR = ''
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
        [dda_peps, dia_peps],
        set_labels=('DDA', 'DIA'),
    )
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/Venn.png')
    plt.close()

def plot_bar(dda_seqs, dia_seqs, dia):
    dia_seqs_q1 = dia_seqs[dia['']]
    dia_seqs_q2 = dia_seqs[dia['']]
    dia_seqs_q3 = dia_seqs[dia['']]

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


def shared_spectra(dda, dia, dda_seqs, dia_seqs, int_peps):

    dda_filtered = dda[dda_seqs.isin(int_peps)].copy()
    dia_filtered = dia[dia_seqs.isin(int_peps)].copy()

    dda_filtered.index = dda_seqs[dda_seqs.isin(int_peps)].values
    dia_filtered.index = dia_seqs[dia_seqs.isin(int_peps)].values

    dda_filtered = dda_filtered.reindex(int_peps)
    dia_filtered = dia_filtered.reindex(int_peps)

    cols = ['Representative_Filename', 'Representative_Scan', 'PSM Charge']

    combined = pd.concat(
        [dda_filtered[cols], dia_filtered[cols]],
        axis=1,
        keys=['dda', 'dia']
    )

    return combined

def cosine(filt_df, file):
    with mgf.read(file) as reader:



def compute_similarity(overlap_df):
    # this is wrong becuase you have to pull from both dda and dia
    files = set(overlap_df['Representative_Filename'])
    for file in files:
        filt_df = overlap_df[overlap_df['Representative_Filename'] == file]
        file = file.split('/')[-1]
        scores = cosine(filt_df, file)



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
    int_peps = np.array(dda_seqs.intersection(dia_seqs))
    overlap_df = shared_spectra(dda_seqs, dia_seqs, dda_seqs, int_peps)

    # extract spectra and compute similarity






