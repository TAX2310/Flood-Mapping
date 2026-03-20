from pathlib import Path
import pandas as pd


def load_metadata(csv_path):
    """
    Load a metadata CSV and remove any unnamed index columns.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df

from pathlib import Path

def get_intersection_ids(s1_csv_path, s2_csv_path, key="tile_id"):
    s1_df = load_metadata(s1_csv_path)
    s2_df = load_metadata(s2_csv_path)

    s1_ids = set(s1_df[key].astype(str))
    s2_ids = set(s2_df[key].astype(str))

    intersection_ids = s1_ids & s2_ids

    # remove .tif using Path.stem
    intersection_ids = {Path(x).stem for x in intersection_ids}

    return intersection_ids
