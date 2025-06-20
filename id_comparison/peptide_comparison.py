import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib_venn import venn2,venn3
from pyteomics import pepxml,mzml,mgf

### paths & params ###
FRAG_SEARCH = "D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/ion.tsv"
MAESTRO_SEARCH_CLUS = 'D:Manish/projects/umpire_data/maestro/HEK/MAESTRO-HEK_w_clustering.tsv'
MAESTRO_SEARCH_NOCLUS = 'D:Manish/projects/umpire_data/maestro/HEK/MAESTRO-HEK_no_clustering.tsv'
SPLIT_GROUPS = 'G1|G2|G3'
MZML_PATHS = [f'D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_0{j}_Q{i}.mzml' for i in range(1, 4) for j in range(1, 4)]
FRAG_PEPXML = [f'D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_0{j}_Q{i}.pepXML' for i in range(1,4) for j in range(1, 4)]
O_PATH = './HEK_2p5ms_Comparison/'
NO_CLUS = True

if not os.path.exists(O_PATH):
    os.makedirs(O_PATH)

### dfs ###
fragpipe = pd.read_csv(FRAG_SEARCH, sep= '\t')
maestro = pd.read_csv(MAESTRO_SEARCH_CLUS, sep = '\t')
if NO_CLUS:
    maestro_nc = pd.read_csv(MAESTRO_SEARCH_NOCLUS, sep = '\t')

### standardize peptides ###

# convert fragpipe to standard pep mod format
# TODO: replace preceding aa character
def standardize_mods(row):
    if row['Assigned Modifications'] is np.nan:
        return row['Peptide Sequence']

    mods = row['Assigned Modifications'].split(',')
    peptide = row['Peptide Sequence']
    for mod in mods:
        mod = mod.strip()
        if 'N-term' in mod:
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', mod)
            peptide = f'[{match.group(1)}]{peptide}'
        else:
            match = re.match(r'^(\d+)([A-Z])\(([-+]?[0-9]*\.?[0-9]+)\)$', mod).groups()
            loc = int(match[0])
            ins = f'({match[1]},{match[2]})'
            index = 0
            i = 0
            while i < loc:
                if peptide[index] == '(':
                    while peptide[index] != ')':
                        index += 1
                    index += 1
                i += 1
                index += 1
            peptide = f'{peptide[:index]}{ins}{peptide[index:]}'

    return peptide

# maestro: get rid of brackets and n and c terminus annots
def process_sequence(row):
    peptide = row['Peptide'][2:-2]
    peptide = peptide.replace('{', '[')
    peptide = peptide.replace('}', ']')
    return peptide

# maestro: handle groups
def split(df, pep):
    filter = df['Default_groups'].str.contains(SPLIT_GROUPS, regex=True)
    return pep[filter]

# standardize pepXML from MSFragger
def standardize_psm_peps(row):
    mods = row['search_hit'][0]['modifications']
    peptide = row['search_hit'][0]['peptide']
    if not mods:
        return peptide

    mod_peptide = peptide
    offset = 0


    for mod in sorted(mods, key=lambda x: x['position']):
        loc = mod['position']
        mass = mod['mass']

        if loc == 0:
            # N-terminal mod: prepend [mass]
            mod_peptide = f'[{mass}]{mod_peptide}'
            offset += len(f'[{mass}]')
        else:

            idx = loc - 1 + offset
            aa = mod_peptide[idx]
            mod_str = f'({aa}, {mass})'
            mod_peptide = mod_peptide[:idx] + mod_str + mod_peptide[idx+1:]
            offset += len(mod_str) - 1  # 1 char removed, mod_str inserted

    return mod_peptide


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

        count = 0
        for pep in frag_maestro_diff:
            if '(' in pep or '[' in pep:
                count += 1
        print(f'% PTM in difference: {count/len(frag_maestro_diff)}')



        # get spectrum ids
        keep_ids = set()
        for psm_path in FRAG_PEPXML:
            psm = pd.DataFrame(pepxml.read(psm_path))
            psm_peps = psm.apply(standardize_psm_peps, axis=1)
            filter = psm_peps.isin(frag_maestro_diff)
            keep_ids.update(set(psm.loc[filter, 'spectrumNativeID']))

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


