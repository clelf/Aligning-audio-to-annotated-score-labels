###################################################################

""" This module contains the main audio-to-annotations functionality and helper functions."""

############################# IMPORTS #############################

import IPython.display as ipd
from libfmp.b import list_to_pitch_activations, plot_chromagram, plot_signal, plot_matrix, \
                     sonify_pitch_activations_with_signal
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

#import sys
#sys.path.insert(0, '../sync_toolbox/synctoolbox/')
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from synctoolbox.feature.csv_tools import read_csv_to_df, df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

import ms3
import libfmp.c2
import os
from Bio import pairwise2 as pw2


############################# GLOBALS #############################

Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6


############################# UTILS #############################

def to_start(quarterbeat):
    """
    Converts "quarterbeat" element from DCML mozart-sontatas corpus from Fraction type to float
    """
    # Note: keep +1 to ensure first note is not attached to 0 even after warping calculation
    return float(quarterbeat.numerator)/float(quarterbeat.denominator)+1


def to_end(start, duration):
    """
    Computes "end" from "start" and "duration"
    """
    return start + duration

def corpus_to_df_musical_time(notes_path):
    """
    Converts notes TSV file as in DCML mozart-sonatas corpus [1] to a dataframe used for warping path synchronization

    Parameters:
        notes_path: str
            Location of the TSV file containing notes information
                Note: the file must at least contain 'quarterbeats', 'duration_qb' and 'midi'

    Returns:
        df_annotation: DataFrame object
            Dataframe with columns ['start', 'duration', 'pitch', 'velocity', 'instrument']
                Note: 'start', 'duration', 'pitch' derive from the original file,
                    'velocity' and 'instrument' are artificially infered
    """
    
    # Load TSV into dataframe
    notes_df = ms3.load_tsv(notes_path) 
    
    # Select and rename columns of interest
    df_annotation = notes_df[['quarterbeats', 'duration_qb', 'midi']].rename(
        columns={'quarterbeats':'start', 'duration_qb':'duration', 'midi': 'pitch'})
    
    # Create "start" column
    df_annotation['start'] = df_annotation['start'].apply(lambda x: to_start(x))
    # Create "end" column
    
    df_annotation['end'] = df_annotation.apply(lambda x: to_end(x['start'], x['duration']), axis=1)
    
    # Re-order columns
    df_annotation = df_annotation[['start', 'duration', 'pitch', 'end']] 
    
    # Infer velocity because needed for synctoolbox processing
    df_annotation['velocity']=1.0 
    
    # Infer instrument, idem
    df_annotation['instrument']="piano"
    
    return df_annotation

def align_corpus_notes_and_labels(notes_path, labels_path):
    """
    Merges notes and labels corpus dataframes together from their TSV file 
    from DCML mozart-sonatas corpus [1], using quarterbeats as the key for an union-based merge.
    
    Parameters:
       notes_path: str
            Location of the TSV file containing notes information
                Note: the file must at least contain 'quarterbeats', 'duration_qb' and 'midi' 
       labels_path: str
            Location of the TSV file containing labels information
                Note: the file must at least contain 'quarterbeats' and 'duration_qb' 
            
    Returns:
        notes_extended: DataFrame object
            DataFrame containing notes and labels information, aligned on notes index     
    """
    
    notes_qb = ms3.load_tsv(notes_path)
    labels_qb = ms3.load_tsv(labels_path)
    
    notes_extended = pd.merge(notes_qb, labels_qb.drop(columns=
                                                        ['duration_qb', 'mc', 'mn', 'mc_onset',
                                                        'mn_onset', 'timesig', 'staff', 'voice']), 
                                left_on=['quarterbeats'], right_on=['quarterbeats'], how='outer')
    
    return notes_extended

def align_warped_notes_labels(df_annotation_warped, notes_labels_extended, mode='compact'):
    """
    After warping path is computed and synchronization of notes dataframe with audio is performed,
    align the labels using notes index as a key. 
        Note: Labels must have been aligned with notes beforehand.
    
    Parameters:
        df_annotation_warped: DataFrame object
            Dataframe resulting from warping process
        notes_labels_extended:DataFrame object
            Dataframe containing corpus information
        mode: str (optional)
            Level of details the result should keep.
            Can take value between: ['compact', 'labels', 'extended', 'scofo']
                Compact: only outputs labels aligned with timestamps
                Labels details: outputs labels and additional label information aligned with timestamps
                Extended: outputs merged notes and labels datasets with additional information, aligned with timestamps
                Sco-fo: outputs notes and additional note information aligned with timestamps [not implemented at the moment]
            Defaults to 'compact'.

    Returns:
        aligned_timestamps_labels: DataFrame object
            Labels and optional information, aligned with timestamps
    """
    
    if mode == 'compact':
        aligned_timestamps_labels = pd.merge(df_annotation_warped.drop(columns= ['velocity',
                                                                                'instrument', 
                                                                                'duration',
                                                                                'pitch',
                                                                                'end']),
                                            notes_labels_extended[['label']], right_index=True, left_index=True)
        aligned_timestamps_labels = aligned_timestamps_labels.dropna(subset='label').drop_duplicates(subset = ['start', 'label'])
        
    elif mode == 'labels':
        
        notes_related_columns = ['staff','voice', 'duration', 'nominal_duration', 'scalar',
                                 'tied','tpc', 'midi', 'chord_id']
        
        # Note: additional columns not present in all pieces could be also added and tested here
        if 'gracenote' in notes_labels_extended:
            notes_related_columns.append('gracenote')
    
        aligned_timestamps_labels = pd.merge(df_annotation_warped.drop(columns=['velocity',
                                                                                'instrument',
                                                                                'pitch']).rename(
            columns={'duration':'duration_time'}),
                                            notes_labels_extended.drop(columns=notes_related_columns),
                                            right_index=True, left_index=True)
        aligned_timestamps_labels = aligned_timestamps_labels.dropna(subset='label').drop_duplicates(subset = ['start', 'label'])

    elif mode == 'extended':
        aligned_timestamps_labels = pd.merge(df_annotation_warped.drop(columns=['velocity',
                                                                                'instrument',
                                                                                'pitch']).rename(
            columns={'duration':'duration_time'}),
                                            notes_labels_extended, right_index=True, left_index=True)
    
    else:
        raise ValueError("'mode' parameter should take either 'compact', 'labels' or 'extended' value")
    
    return aligned_timestamps_labels


def get_features_from_audio(audio, tuning_offset, Fs, feature_rate, visualize=False):
    """ 
    Adapted from synctoolbox [2] tutorial: `sync_audio_score_full.ipynb`
    
    Takes as input the audio file and computes quantized chroma and DLNCO features.
    
    Parameters:
        audio: str
            Path to file in .wav format
        tuning_offset: int
            Tuning offset used to shift the filterbank (in cents)
        Fs: float
            Sampling rate of f_audio (in Hz)
        feature_rate: int
            Features per second
        visualize: bool (optional)
            Whether to visualize chromagram of audio quantized chroma features. Defaults to False.
            
    Returns:
    f_chroma_quantized: np.ndarray
        Quantized chroma representation
    f_DLNCO : np.array
        Decaying locally adaptive normalized chroma onset (DLNCO) features
    """
    
    f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate, verbose=visualize)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    
    if visualize:
        plot_chromagram(f_chroma_quantized, title='Quantized chroma features - Audio', Fs=feature_rate, figsize=(9,3))

    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate, feature_sequence_length=f_chroma_quantized.shape[1], visualize=visualize)
    
    return f_chroma_quantized, f_DLNCO


def get_features_from_annotation(df_annotation, feature_rate, visualize=False):
    """ 
    Adapted from synctoolbox [2] tutorial: `sync_audio_score_full.ipynb`
    
    Takes as input symbolic annotations dataframe and computes quantized chroma and DLNCO features.
    
    Parameters:
        df_annotation: DataFrame object
            Dataframe of notes annotations containing ['start', 'duration', 'pitch', 'velocity', 'instrument']        
        feature_rate: int
            Features per second
        visualize: bool (optional)
            Whether to visualize chromagram of audio quantized chroma features. Defaults to False.
            
    Returns:
    f_chroma_quantized: np.ndarray
        Quantized chroma representation
    f_DLNCO : np.array
        Decaying locally adaptive normalized chroma onset (DLNCO) features
    """
       
    f_pitch = df_to_pitch_features(df_annotation, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    
    if visualize:
        plot_chromagram(f_chroma_quantized, title='Quantized chroma features - Annotation', Fs=feature_rate, figsize=(9, 3))
    f_pitch_onset = df_to_pitch_onset_features(df_annotation)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset,
                                            feature_rate=feature_rate,
                                            feature_sequence_length=f_chroma_quantized.shape[1],
                                            visualize=visualize)
    
    return f_chroma_quantized, f_DLNCO

def warp_annotations(df_annotation, warping_path, feature_rate):
    """ 
    Adapted from synctoolbox [2] tutorial: `sync_audio_score_full.ipynb`
    
    Warp timestamps to annotations after having computed the warping path between audio and annotations.

    Parameters:
        df_annotation: DataFrame object
            Dataframe of notes annotations containing ['start', 'duration', 'pitch', 'velocity', 'instrument']    
        warping_path: np.ndarray(2:)
            Warping path
        feature_rate: int
            Features per second

    Returns:
        Notes annotations warped with corresponding timestamps
    """
    
    df_annotation_warped = df_annotation.copy(deep=True)
    df_annotation_warped["end"] = df_annotation_warped["start"] + df_annotation_warped["duration"]
    df_annotation_warped[['start', 'end']] = scipy.interpolate.interp1d(warping_path[1] / feature_rate, 
                               warping_path[0] / feature_rate, kind='linear', fill_value="extrapolate")(df_annotation[['start', 'end']])
    df_annotation_warped["duration"] = df_annotation_warped["end"] - df_annotation_warped["start"]
    
    return df_annotation_warped




def evaluate_matching(df_original_notes, df_warped_notes, verbose=True):
    """ 
    Evaluates matching of original corpus notes list with warped notes.
    If DTW runs correctly, all notes should have been matched.
    
    Further evaluation can be performed if groundtruth annotations of the audio are known,
    with synctoolbox function `evaluate_synchronized_positions`.
    
    Parameters:
        df_original_notes (DataFrame object): original notes list from corpus
        df_warped_notes (DataFrame object): 
        verbose (bool, optional): Prints evaluation. Defaults to True.

    Returns:
        Matching score, i.e. percentage of original notes that have found a match
    """
    
    global_align = pw2.align.globalxx(list(df_original_notes.pitch.apply(str)),
                                      list(df_warped_notes.pitch.apply(str)), gap_char = ['-'])
    matching_score = global_align[0][2]/len(df_original_notes)
    if verbose:
        print("Matching percentage: {:.4}%".format(matching_score*100))
        print("Number of unmachted notes: {:}".format(len(df_original_notes)-global_align[0][2]))

    return matching_score




def align_notes_labels_audio(notes_path, labels_path, audio_path,
                             store=True, store_path=os.path.join(os.getcwd(), 'alignment_results', 'result.csv'),
                             verbose=False, visualize=False, evaluate=False, mode='compact'):
    """
    This function performs the whole pipeline of aligning an audio recording of a piece and its
    corresponding labels annotations from DCML's Mozart sonatas corpus [1], using synctoolbox dynamic
    time warping (DTW) tools [2]. It takes as input the paths to the audio file, to the labels TSV file
    and the notes TSV file as well, for DTW to align single notes events specifically. It returns a
    dataframe containing labels and their corresponding timestamps within the audio.
    
    Depending on the `mode` passed, the result can contain the minimal information (labels and timestamps)
    that are useful for live visualization (e.g. with SonicVisualiser) or more detailed information about labels
    (and possibly notes) as in the original TSV files.

    Parameters:
        notes_path: str
            Location of the TSV file containing notes information
                Note: the file must at least contain 'quarterbeats', 'duration_qb' and 'midi'
        labels_path: str
            Location of the TSV file containing labels information
                Note: the file must at least contain 'quarterbeats' and 'duration_qb'
        audio_path: str
            Location of the audio file. The audio should be in .wav format.
        store: bool (optional)
            Stores the alignment result. Defaults to False.
        store_path: str (optional)
            If store is set to True, path to store the result in. Defaults to "alignment_results/result.csv".
        verbose: bool (optional)
            Prints information. Defaults to False.
        visualize: bool (optional)
            Prints DTW process visualizations, to be used if function called in notebook for example.
            Defaults to False.
        evaluate: bool (optional)
            Prints DTW matching score. Defaults to False.
        mode: str (optional)
            Level of details the result should keep. 
            Can take value between: ['compact', 'labels', 'extended']
                Compact: only outputs labels aligned with timestamps
                Labels details: outputs labels and additional label information aligned with timestamps
                Extended: outputs merged notes and labels datasets with additional information, aligned with timestamps                
            Defaults to 'compact'.

    Returns:
        result: DataFrame object
            Dataframe containing labels and their corresponding timestamps within the audio.
            
            
    References: 
    [1] Hentschel, J., Neuwirth, M. and Rohrmeier, M., 2021. 
        The Annotated Mozart Sonatas: Score, Harmony, and Cadence. 
        Transactions of the International Society for Music Information Retrieval, 4(1), pp.67–80. 
        DOI: http://doi.org/10.5334/tismir.63
        [https://github.com/DCMLab/mozart_piano_sonatas]
     
    [2] Meinard Müller, Yigitcan Özer, Michael Krause, Thomas Prätzlich, and Jonathan Driedger. and Frank Zalkow. 
        Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization. 
        Journal of Open Source Software (JOSS), 6(64), 2021.
        [https://github.com/meinardmueller/synctoolbox]
    
    """
    
    # Prepare annotation format
    df_annotation = corpus_to_df_musical_time(notes_path)
    # Keep track of notes annotations and labels correspondances
    df_annotation_extended = align_corpus_notes_and_labels(notes_path, labels_path)
    
    # Load audio
    audio, _ = librosa.load(audio_path, Fs)

    # Estimate tuning
    # Alternative: librosa.estimate_tuning
    tuning_offset = estimate_tuning(audio, Fs)
    if verbose:
        print('Estimated tuning deviation for recording: %d cents' % (tuning_offset))
    
    # Compute features from audio
    f_chroma_quantized_audio, f_DLNCO_audio = get_features_from_audio(audio, tuning_offset, Fs, feature_rate, visualize=visualize)
    
    # Compute features from annotations
    f_chroma_quantized_annotation, f_DLNCO_annotation = get_features_from_annotation(df_annotation, feature_rate, visualize=visualize)
    
    # Calculate chroma shift between audio and annotations
    f_cens_1hz_audio = quantized_chroma_to_CENS(f_chroma_quantized_audio, 201, 50, feature_rate)[0]
    f_cens_1hz_annotation = quantized_chroma_to_CENS(f_chroma_quantized_annotation, 201, 50, feature_rate)[0]
    opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_audio, f_cens_1hz_annotation)
    
    if verbose:
        print('Pitch shift between the audio recording and score, determined by DTW:', opt_chroma_shift, 'bins')

    # Apply potential shift to audio and annotations features
    f_chroma_quantized_annotation = shift_chroma_vectors(f_chroma_quantized_annotation, opt_chroma_shift)
    f_DLNCO_annotation = shift_chroma_vectors(f_DLNCO_annotation, opt_chroma_shift)
    
    # Compute warping path
    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_audio, 
                      #f_onset1=f_DLNCO_audio, 
                      f_chroma2=f_chroma_quantized_annotation, 
                      #f_onset2=f_DLNCO_annotation, 
                      input_feature_rate=feature_rate, 
                      step_weights=step_weights, 
                      threshold_rec=threshold_rec, 
                      verbose=visualize)
    
    # Make warping path monotonic
    wp = make_path_strictly_monotonic(wp)
    
    # Use warping path to align annotations to real timestamps 
    df_annotation_warped = warp_annotations(df_annotation, wp, feature_rate)
    
    # Evaluate matching score after warping
    if evaluate:
        _ = evaluate_matching(df_annotation, df_annotation_warped, verbose=True)
    
    # Align time-aligned annotations of notes with labels
    result = align_warped_notes_labels(df_annotation_warped, df_annotation_extended, mode)
    
    # Store
    if store:
        result.to_csv(store_path, index = False)
    
    return result