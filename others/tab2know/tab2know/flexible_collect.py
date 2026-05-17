"""
Flexible metadata collection for tab2know: supports multiple directory layouts
(basedir/csv, basedir/, basedir/csvs, basedir/tables, basedir/data) instead of
assuming basedir/csv only. Uses get_file_meta from collect; BASE URI matches
the rest of tab2know (cs.vu.nl/tab2know).
"""
import os, sys, csv, json
try:
    from .collect import get_file_meta
except ImportError:
    from collect import get_file_meta

BASE = "http://cs.vu.nl/tab2know/"

def find_csv_directory(basedir):
    """
    Find CSV directory under basedir, trying multiple layouts:
    1. basedir/csv/ (original layout)
    2. basedir/ (CSV files directly in basedir)
    3. basedir/csvs/
    4. basedir/tables/
    5. basedir/data/
    """
    possible_dirs = [
        os.path.join(basedir, 'csv'),      # original layout
        basedir,                           # CSV files directly in basedir
        os.path.join(basedir, 'csvs'),
        os.path.join(basedir, 'tables'),
        os.path.join(basedir, 'data'),
    ]

    for csv_dir in possible_dirs:
        if os.path.exists(csv_dir) and os.path.isdir(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                return csv_dir

    # Fallback for backward compatibility
    print(f"Warning: no CSV directory found, using default: {os.path.join(basedir, 'csv')}")
    return os.path.join(basedir, 'csv')

def get_all_metadata_flexible(basedir, paper_prefix=''):
    """
    Flexible get_all_metadata: discovers CSV directory automatically.
    """
    csv_dir = find_csv_directory(basedir)

    f = os.path.join(basedir, 'id_lookup.txt')
    if not os.path.isfile(f):
        id_lookup = {}
    else:
        id_lookup = {
            a: b
            for line in open(f) for a, b in [line.strip().split()]
        }

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Warning: no CSV files in {csv_dir}")
        return

    for name in csv_files:
        fname = os.path.join(csv_dir, name)
        if os.path.isfile(fname):
            yield get_file_meta(fname,
                                paper_prefix=paper_prefix,
                                id_lookup=id_lookup)

def get_csv_count_flexible(basedir):
    """
    Count CSV files using the same flexible directory lookup.
    """
    csv_dir = find_csv_directory(basedir)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    return len(csv_files)
