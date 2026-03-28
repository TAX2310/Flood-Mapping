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
#tile_id
from pathlib import Path

def get_intersection_ids(s1_csv_path, s2_csv_path,
                         floodmap_key="floodmap_id",
                         tile_key="tile_id"):

    s1_df = load_metadata(s1_csv_path)
    s2_df = load_metadata(s2_csv_path)

    # Step 1: find matching floodmap_ids
    s1_flood_ids = set(s1_df[floodmap_key].astype(str))
    s2_flood_ids = set(s2_df[floodmap_key].astype(str))

    matching_flood_ids = s1_flood_ids & s2_flood_ids

    # Step 2: filter both datasets to matching floodmap_ids
    s1_filtered = s1_df[s1_df[floodmap_key].astype(str).isin(matching_flood_ids)]
    s2_filtered = s2_df[s2_df[floodmap_key].astype(str).isin(matching_flood_ids)]

    # Step 3: extract tile_ids from both
    s1_tiles = set(s1_filtered[tile_key].astype(str))
    s2_tiles = set(s2_filtered[tile_key].astype(str))

    # Step 4: get intersection of tile_ids
    matching_tiles = s1_tiles & s2_tiles

    # Optional: remove .tif
    matching_tiles = {Path(x).stem for x in matching_tiles}

    return matching_tiles

from pathlib import Path
import pandas as pd


def get_valid_matching_tile_ids(
    s1_csv_path,
    s2_csv_path,
    floodmap_key="floodmap_id",
    tile_key="tile_id",
    floodmap_date_col="floodmap_date",
    sentinel_time_col="sentinel_timestamp",
    max_time_diff_hours=24,
    require_matching_floodmap_id=True,
    strip_suffix=True,
):
    """
    Return tile_ids that exist in both S1 and S2, where each row is internally
    valid based on the time difference between floodmap_date and sentinel_timestamp.

    Parameters
    ----------
    s1_csv_path, s2_csv_path : str or Path
        Paths to the S1 and S2 metadata CSVs.
    floodmap_key : str
        Column used for floodmap identifier.
    tile_key : str
        Column used for tile identifier.
    floodmap_date_col : str
        Column containing flood map timestamp.
    sentinel_time_col : str
        Column containing image acquisition timestamp.
    max_time_diff_hours : float
        Maximum allowed absolute difference in hours between floodmap_date and
        sentinel_timestamp within each dataset.
    require_matching_floodmap_id : bool
        If True, require matches on both floodmap_id and tile_id.
        If False, match only on tile_id after temporal filtering.
    strip_suffix : bool
        If True, remove file suffix using Path.stem.

    Returns
    -------
    set[str]
        Matching tile_ids that pass the temporal validity filter.
    """

    s1_df = load_metadata(s1_csv_path).copy()
    s2_df = load_metadata(s2_csv_path).copy()

    # Parse timestamps
    s1_df[floodmap_date_col] = pd.to_datetime(s1_df[floodmap_date_col], errors="coerce")
    s1_df[sentinel_time_col] = pd.to_datetime(s1_df[sentinel_time_col], errors="coerce")

    s2_df[floodmap_date_col] = pd.to_datetime(s2_df[floodmap_date_col], errors="coerce")
    s2_df[sentinel_time_col] = pd.to_datetime(s2_df[sentinel_time_col], errors="coerce")

    # Compute absolute time difference in hours within each dataset
    s1_df["time_diff_hours"] = (
        (s1_df[floodmap_date_col] - s1_df[sentinel_time_col]).abs().dt.total_seconds() / 3600
    )
    s2_df["time_diff_hours"] = (
        (s2_df[floodmap_date_col] - s2_df[sentinel_time_col]).abs().dt.total_seconds() / 3600
    )

    # Keep only rows within acceptable internal range
    s1_valid = s1_df[s1_df["time_diff_hours"] <= max_time_diff_hours].copy()
    s2_valid = s2_df[s2_df["time_diff_hours"] <= max_time_diff_hours].copy()

    # Normalise join columns as strings
    s1_valid[tile_key] = s1_valid[tile_key].astype(str)
    s2_valid[tile_key] = s2_valid[tile_key].astype(str)

    if require_matching_floodmap_id:
        s1_valid[floodmap_key] = s1_valid[floodmap_key].astype(str)
        s2_valid[floodmap_key] = s2_valid[floodmap_key].astype(str)

        merged = s1_valid.merge(
            s2_valid,
            on=[floodmap_key, tile_key],
            suffixes=("_s1", "_s2"),
        )
    else:
        merged = s1_valid.merge(
            s2_valid,
            on=[tile_key],
            suffixes=("_s1", "_s2"),
        )

    tile_ids = set(merged[tile_key])

    if strip_suffix:
        tile_ids = {Path(x).stem for x in tile_ids}

    return tile_ids
