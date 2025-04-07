# imports
import argparse
import os.path
from typing import Optional, Literal

import ms3.cli
import pandas as pd
from tqdm.auto import tqdm

from utils import align_notes_labels_audio


def batch_process(
        csv_path: Optional[str] = None,
        store: bool = True,
        store_path: Optional[str] = None,
        verbose: bool = False,
        visualize: bool = False,
        evaluate: bool = False,
        mode: Literal['compact', 'labels', 'extended'] = 'compact'
):
    """

    Args:
        csv_path:
            Path to a CSV file specifying paths for batch processing. The file needs to contain at
            least the columns "audio" and "notes" and may come with an additional column "labels".
            All values must be absolute paths or relative paths that resolve correctly for the
            current working directory. If, in addition, you want to specify output filenames (with
            or without .csv/.tsv extension), include them in a column called "name".
        store:
        store_path:
        verbose:
        visualize:
        evaluate:
        mode:
    """
    if store_path:
        assert os.path.isdir(store_path), (f"store_path arg needs to be an existing directory for "
                                               f"batch processing. Got {store_path!r}")
    mapping = pd.read_csv(csv_path)
    assert all(
        col in mapping.columns
        for col in ("audio", "notes")
    ), (f"CSV file needs to have at least the columns "
        f"'audio' and 'notes' containing file paths. The file "
        f"{csv_path!r} comes with the "
        f"columns {mapping.columns!r}.")
    has_labels = "labels" in mapping.columns
    has_names = "name" in mapping.columns
    batch_size = len(mapping)
    for row in tqdm(mapping.itertuples(), total=batch_size):
        if has_names and not pd.isnull(row.name):
            output_path = row.name
            fname, ext = os.path.splitext(output_path)
            if ext not in (".csv", ".tsv"):
                output_path = fname + ".tsv"
            if store_path:
                output_path = os.path.join(store_path, output_path)
        else:
            output_path = store_path
        align_notes_labels_audio(
            audio_path=row.audio,
            notes_path=row.notes,
            labels_path=None if not has_labels else row.labels,
            store=store,
            store_path=output_path,
            verbose=verbose,
            visualize=visualize,
            evaluate=evaluate,
            mode=mode
        )

def main(
        audio_path: Optional[str] = None,
        notes_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        store: bool = True,
        store_path: Optional[str] = None,
        verbose: bool = False,
        visualize: bool = False,
        evaluate: bool = False,
        mode: Literal['compact', 'labels', 'extended'] = 'compact'
):
    if csv_path:
        batch_process(
            csv_path=csv_path,
            store=store,
            store_path=store_path,
            verbose=verbose,
            visualize=visualize,
            evaluate=evaluate,
            mode=mode
        )
    else:
        align_notes_labels_audio(
            audio_path=audio_path,
            notes_path=notes_path,
            labels_path=labels_path,
            store=store,
            store_path=store_path,
            verbose=verbose,
            visualize=visualize,
            evaluate=evaluate,
            mode=mode
        )


def parse_args():
    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    
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
        
    """
    )
    parser.add_argument(
        '-c', '--csv',
        help='Path to a CSV file specifying paths for batch processing. The file needs to contain at least the '
             'columns "audio" and "notes" and may come with an additional column "labels". All values must be '
             'absolute paths or relative paths that resolve correctly for the current working directory. If, '
             'in addition, you want to specify output filenames (with or without .csv/.tsv extension), include '
             'them in a column called "name"'
             'When you specify this argument, -a, -n, and -l will be ignored.', )
    parser.add_argument('-a', '--audio', help='Path to audiofile')
    parser.add_argument('-n', '--notes', help='Path to the notes TSV file')
    parser.add_argument('-l', '--labels', help='Path to the labels TSV file')
    parser.add_argument(
        '-o', '--output',
        help='Folder or filepath (.csv/.tsv) for storing the alignment result. Can be relative, defaults to current '
             'working directory.'
    )
    parser.add_argument(
        '-m', '--mode',
        help="Output format mode, to choose between ['compact', 'labels', 'extended', 'scofo']. default: ",
        default='compact'
    )
    parser.add_argument(
        '-e', '--evaluate', help="Evaluate warping mode. default: False", action='store_true', default=False
    )
    args = parser.parse_args()
    if args.csv:
        if args.output:
            args.output = ms3.cli.check_and_create(args.output)
        for arg in ("audio", "notes", "labels"):
            if args.__dict__[arg]:
                print(f"--{arg} is ignored when --csv is specified.")
    else:
        assert args.audio and args.notes, f"You need to either specify --csv or both --audio and --notes."
        if args.output:
            args.output = ms3.resolve_dir(args.output)
    return args


def run():
    """ Audio-to-annotations aligner """
    args = parse_args()
    main(
        audio_path=args.audio,
        notes_path=args.notes,
        labels_path=args.labels,
        csv_path=args.csv,
        store=True,
        store_path=args.output,
        verbose=False,
        visualize=False,
        evaluate=args.evaluate,
        mode=args.mode
    )

if __name__ == '__main__':
    run()
