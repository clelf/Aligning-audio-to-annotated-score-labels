# imports
import os, argparse
from utils import align_notes_labels_audio
# from ms3 import check_dir



def main():
    """ Audio-to-annotations aligner """
    
    # define parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    
    This script performs the alignment between an audio recording of a piece present in DCML Mozart sonatas 
    corpus [1], using synctoolbox dynamic time warping (DTW) tools [2]. 

    First, download the corpus via https://github.com/DCMLab/mozart_piano_sonatas.
    Second, navigate to the top level of the mozart_piano_sonatas repository and generate the notes and labels files
    needed for audio-to-annotation alignment by running:

    >>> python ms3 extract -N [folder_to_write_notes_file_to] -X [folder_to_write_labels_file_to] -q

    This will provide additional quarterbeats information needed for alignment.

    Once the files locations are all identified, you can run:
    >>> python aligner.py -a [audio_WAV_file] -n [notes_TSV_file] -l [labels_TSV_file] -o [CSV_file_to_write_results_to]

    This default command line will store a CSV file with minimal information i.e. labels and corresponding timestamps,
    useful to visualize (e.g. with SonicVisualiser).
        

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
        
    """)
    
    parser.add_argument('-a', '--audio', help='Path to audiofile', required=True)
    #parser.add_argument('-p','--piece', help='Name of the desired piece from the Mozart corpus')
    parser.add_argument('-n', '--notes', help='Path to the notes TSV file', required=True)
    parser.add_argument('-l', '--labels', help='Path to the labels TSV file', required=True)
    #parser.add_argument('-o', '--output', type=check_dir, default=os.path.join(os.getcwd(), 'alignment'), help='Folder for storing the alignment result. Can be relative, defaults to ./alignment')
    parser.add_argument('-o', '--output', default=os.path.join(os.getcwd(), 'alignment_results', 'result.csv'), help='Folder for storing the alignment result. Can be relative, defaults to ./alignment_results')
    parser.add_argument('-m', '--mode', help="Output format mode, to choose between ['compact', 'labels', 'extended', 'scofo']. default: ", default='compact')
    parser.add_argument('-e', '--evaluate', help="Evaluate warping mode. default: False", action='store_true', default=False)

    
    args = parser.parse_args()
    
    output_dir = args.output
    audio_path = args.audio
    notes_path = args.notes
    labels_path = args.labels
    mode = args.mode
    evaluate = args.evaluate
    
    
    align_notes_labels_audio(notes_path, labels_path, audio_path, 
                             store=True, store_path=output_dir,
                             verbose=False, visualize=False, evaluate=evaluate, mode=mode)
    

    

if __name__ == '__main__':
    main()
