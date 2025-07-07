import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib_venn import venn2,venn3
from pyteomics import pepxml,mzml,mgf

### paths & params ###
FRAG_SEARCH = "E:/Users/Manish/astral/astral_data/HEK/fragpipe/3p5ms/ion.tsv"
MAESTRO_SEARCH_CLUS = 'D:Manish/projects/umpire_data/maestro/HEK/dia/2p5ms/Maestro-HEK_DIA_Clustering-identified_variants_merged_protein_regions-main.tsv'
MAESTRO_SEARCH_NOCLUS = 'D:Manish/projects/umpire_data/maestro/HEK/dia/2p5ms/Maestro-HEK_DIA_No_Clustering-identified_variants_merged_protein_regions-main.tsv'
SPLIT_GROUPS = 'G4|G5|G6'
MZML_PATHS = [f'E:/Users/Manish/astral/astral_data/HEK/fragpipe/3p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_3p5ms_0p5_2Th_0{j}_Q{i}.mzml' for i in range(1, 4) for j in range(1, 4)]
FRAG_PEPXML = [f'E:/Users/Manish/astral/astral_data/HEK/fragpipe/3p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_3p5ms_0p5_2Th_0{j}_Q{i}.pepXML' for i in range(1,4) for j in range(1, 4)]
O_PATH = 'D:Manish/projects/umpire_data/HEK_3p5ms_Fragtide_Comparison/'
NO_CLUS = True

if not os.path.exists(O_PATH):
    os.makedirs(O_PATH)

### dfs ###
fragpipe = pd.read_csv(FRAG_SEARCH, sep= '\t')
maestro = pd.read_csv(MAESTRO_SEARCH_CLUS, sep = '\t')
if NO_CLUS:
    maestro_nc = pd.read_csv(MAESTRO_SEARCH_NOCLUS, sep = '\t')

### standardize peptides ###

# convert fragpipe unmodified pep + parent mass
# unmodified peptide comparison
def standardize_mods(row):
    peptide = row['Peptide Sequence']
    parent_mass = int(row['Observed Mass'])
    return f'{peptide}'

# maestro: get rid of brackets and n and c terminus annots
def process_sequence(row):
    peptide = str(row['Unmodified_sequence']).strip('.')
    parent_mass = int(row['Parent_mass_for_variant'])
    return f'{peptide}'

# maestro: handle groups
def split(df, pep):
    filter = df['Default_groups'].str.contains(SPLIT_GROUPS, regex=True)
    return pep[filter]

# standardize pepXML from MSFragger
def standardize_psm_peps(row):
    peptide = row['search_hit'][0]['peptide']
    parent_mass = int(row['precursor_neutral_mass'])

    return f'{peptide}'


### generate venn diagrams ###
def plot_venn2(maestro_peps, frag_peps):
    plt.figure(figsize=(6, 6))
    venn2(
        [maestro_peps, frag_peps],
        set_labels=('Maestro', 'FragPipe'),
    )
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/Venn.png')
    plt.close()

def plot_venn3(maestro_peps, maestro_nc_peps, frag_peps):
    plt.figure(figsize=(6, 6))  # create a figure with a set size
    venn3(
        [maestro_peps, maestro_nc_peps, frag_peps],
        set_labels=('Maestro Cluster', 'Maestro No Cluster', 'FragPipe')
    )
    plt.tight_layout()
    plt.savefig(f'{O_PATH}/Venn.png')
    plt.close()

def plot_bar(q1_nums, q2_nums, q3_nums):
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


### write set difference to mgf ###

def select_spectra(input_spectra, allowed_ids):
    for spec in input_spectra:
        spec_id = spec.get('id')
        if spec_id in allowed_ids:
            mz_array = spec['m/z array']
            intensity_array = spec['intensity array']
            precursor = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
            pepmass = precursor['selected ion m/z']
            charge = int(precursor.get('charge state', 1))
            rt = spec['scanList']['scan'][0]['scan start time']

            yield {
                'm/z array': mz_array,
                'intensity array': intensity_array,
                'params': {
                    'TITLE': spec_id,
                    'PEPMASS': [pepmass],
                    'CHARGE': f'{charge}+',
                    'RTINSECONDS': rt
                }
            }

# Combine spectra from all mzMLs
def combined_selected_spectra(mzml_files, keep_set):
    for file in mzml_files:
        with mzml.MzML(file) as reader:
            yield from select_spectra(reader, keep_set)

def main():
    if NO_CLUS:

        # only use peptides from the MSFragger ids
        frag_peps = set(fragpipe.apply(standardize_mods, axis=1))
        maestro_peps = maestro.apply(process_sequence, axis=1)
        maestro_peps = set(split(maestro, maestro_peps))
        maestro_no_clus_peps = maestro_nc.apply(process_sequence, axis=1)
        maestro_no_clus_peps = set(split(maestro_nc, maestro_no_clus_peps))

        # generate Venn 3
        plot_venn3(maestro_peps, maestro_no_clus_peps, frag_peps)

        # union, difference
        maestro_union = maestro_peps.union(maestro_no_clus_peps)
        frag_maestro_diff = frag_peps.difference(maestro_union)
        print(f'number of exclusive peptides: {len(frag_maestro_diff)}')


        # get spectrum ids
        keep_ids = set()
        count_q1 = set()
        count_q2 = set()
        count_q3 = set()
        for psm_path in FRAG_PEPXML:
            psm = pd.DataFrame(pepxml.read(psm_path))
            psm_peps = psm.apply(standardize_psm_peps, axis=1)
            filter = psm_peps.isin(frag_maestro_diff)
            keep_ids.update(set(psm.loc[filter, 'spectrumNativeID']))
            if 'Q1' in psm_path:
                count_q1.update(hit[0]['peptide'] for hit in psm.loc[filter, 'search_hit'] if hit)
            elif 'Q2' in psm_path:
                count_q2.update(hit[0]['peptide'] for hit in psm.loc[filter, 'search_hit'] if hit)
            else:
                count_q3.update(hit[0]['peptide'] for hit in psm.loc[filter, 'search_hit'] if hit)

        plot_bar(len(count_q1), len(count_q2), len(count_q3))

        # write spectra
        mgf.write(combined_selected_spectra(MZML_PATHS, keep_ids), f'{O_PATH}/subset.mgf')

    else:
        frag_peps = set(frag_df.apply(standardize_mods, axis=1))
        maestro_peps = maestro.apply(process_sequence, axis=1)
        maestro_peps = set(maestro_peps.apply(split, axis=1))

        # generate Venn 2
        plot_venn2(maestro_peps, frag_peps)

        # difference
        frag_maestro_diff = frag_peps.difference(maestro_peps)
        psm_peps = psm.apply(standardize_psm_peps, axis=1)

        filter = psm_peps.isin(frag_maestro_diff)
        keep_ids = set(psm.loc[filter, 'spectrumNativeID'])

        mgf.write(combined_selected_spectra(MZML_PATHS, keep_ids), f'{O_PATH}/subset.mgf')

    return 0



if __name__ == "__main__":
    main()


