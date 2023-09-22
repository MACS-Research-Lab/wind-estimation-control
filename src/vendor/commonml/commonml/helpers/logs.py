from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def get_tensorboard_scalar_frame(path: str) -> pd.DataFrame:
    """
    Get a pandas dataframe containing scalar values logged by tensorboard.

    Parameters
    ----------
    path : str
        Path to directory containing tensorboard log files.

    Returns
    -------
    pd.DataFrame
    """
    ea = event_accumulator.EventAccumulator(
        path=path,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    columns = ea.Tags().get('scalars', [])
    columns_split = [c.split('/') for c in columns]
    columns_split = list(zip(*columns_split))
    multiindex = pd.MultiIndex.from_arrays(columns_split)

    dfs = []
    for column in columns:
        records = ea.Scalars(column)
        df = pd.DataFrame(records)
        df.set_index('step', drop=True, inplace=True)
        df.drop(columns=['wall_time'], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, axis='columns')

    return pd.DataFrame(df.values, columns=multiindex, index=df.index)