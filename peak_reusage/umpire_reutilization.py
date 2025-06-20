import pandas as pd
from pyteomics import mzml

### Parameters
# TODO: use argparse
TRACE_PATH = '''D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_01_PeakCluster.csv'''
Q1_PATH = '''D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_01_Q1.mzML'''
Q2_PATH = '''D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_01_Q2.mzML'''
Q3_PATH = '''D:/Manish/projects/umpire_data/dia_extracted/astral_extracted_fragpipe/HEK/2p5ms/20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_01_Q3.mzML'''

### Helpers
def ms2_assignment(pseudo_path, traces):
    pseudo = mzml.read(pseudo_path)
    peaks = pd.Series([None] * len(traces), index=traces.index, dtype=object)

    # output to console
    num_spectra = 0
    num_no_prec = 0
    num_non_unique = 0
    num_traces_non_unique = 0

    # total peaks in ms2
    total_count = 0

    for spectra in pseudo:
        num_spectra = num_spectra + 1

        # lookup varaibles
        rt = spectra['scanList']['scan'][0]['scan start time'] / 60
        ion_mz = spectra['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
        charge = spectra['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']

        filter = (traces['mz1'] == ion_mz) & (traces['StartRT'] <= rt) & (traces['EndRT'] >= rt) & (
                    traces['Charge'] == charge)

        lookup = filter.sum()
        peak = dict(zip(spectra['m/z array'], spectra['intensity array']))
        if lookup == 0:
            num_no_prec += 1
            continue
        elif lookup > 1:
            num_non_unique += 1
            true_indices = filter[filter].index
            filter = pd.Series([False for i in range(len(filter))], index=filter, dtype=bool)
            filter.iloc[true_indices[0]] = True
            num_traces_non_unique += len(true_indices)
            peaks.iloc[filter] = [peak]
        else:
            peaks.loc[filter] = [peak]

        total_count += len(spectra['m/z array'])

    print(f'Processing {pseudo_path}')
    print(f'number of extracted spectra: {num_spectra}')
    print(f'number of spectra without precursors: {num_no_prec}')
    print(f'number of non-unique spectra: {num_non_unique}')
    print(f'number of spectra found in traces: {num_spectra - num_no_prec}')
    print(f'number of non-unique traces that map to spectra: {num_traces_non_unique}')
    print('\n')

    return peaks, total_count

def resued_peaks(trace, peak_set):
    # row level peak sharing helper
    def find_peaks(db_peaks, peaks):
        if db_peaks is None:
            return None

        reused = dict()
        for key, value in peaks.items():
            if key in db_peaks:
                if db_peaks[key] == value:
                    reused[key] = value

        if len(reused) == 0:
            return None

        return reused

    matched_peaks = pd.Series([None] * len(peak_set), index=peak_set.index, dtype=object)
    for i in range(len(trace)):

        # trace does not have associated MS2
        peaks = peak_set.iloc[i]
        if peaks is None:
            continue

        # information about the current peak
        mz1 = trace['mz1'].iloc[i]
        rt_start = trace['StartRT'].iloc[i]
        rt_end = trace['EndRT'].iloc[i]

        # filter excluding self
        filter = ((abs(trace['mz1'] - mz1) < 2) & (
                    ((trace['StartRT'] < rt_start) & (trace['EndRT'] > rt_start)) | (
                        (trace['StartRT'] < rt_end) & (trace['EndRT'] > rt_end))))
        if filter.sum() == 0:
            continue
        filter.iloc[i] = False
        filter_df = trace[filter].copy()
        filter_peaks = peak_set[filter].copy()

        # finding reused peaks
        reused_peaks = {}
        for j in range(len(filter_df)):
            reused = find_peaks(filter_peaks.iloc[j], peaks)
            if reused:
                reused_peaks[filter_df.iloc[j]['mz1']] = reused

        if reused_peaks:
            matched_peaks.iloc[i] = [reused_peaks]

    return matched_peaks

# TODO: Need a robust way to count reutilization
def count_reused_peaks(matched_set, masses):
    match_filter = matched_set.notna()
    matched_peaks = matched_set[match_filter]

    count = 0
    visited = set()
    for mp, mz1 in zip(matched_peaks, masses):
        m = mp[0]
        for key, value in m.items():
            if key not in visited:
                count += len(m[key])
                visited.add(key)
        visited.add(mz1)

    return count


def main():

    # Read in Peak Traces
    trace_df = pd.read_csv(TRACE_PATH)
    print(f'Number of Peak Traces: {len(trace_df)}')

    # Assign MS2 to Traces
    peaks = []
    total_count = []
    for path in [Q1_PATH, Q2_PATH, Q3_PATH]:
        assignment = ms2_assignment(path, trace_df)
        peaks.append(assignment[0])
        total_count.append(assignment[1])

    # Sanity Check
    for i in [1, 2, 3]:
        peaks[i-1].to_csv(f'peaks_Q{i}.csv', index=False)

    # Finding Matched Peaks
    reused_peaks = []
    for p in peaks:
        reused_peaks.append(resued_peaks(trace_df, p))

    # Counting
    for i in range(len(reused_peaks)):
        print(f'Results for Q{i+1}')
        reused_peaks_count = count_reused_peaks(reused_peaks[i], trace_df['mz1'])
        print(f'reused peaks: {reused_peaks_count}')
        print(f'total peaks: {total_count[i]}')
        reutilization_rates = reused_peaks_count / total_count[i]
        print(f'reutilization rate: {reutilization_rates}')

    return 0

if __name__ == "__main__":
    main()








