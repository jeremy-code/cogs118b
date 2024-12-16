"""This module preprocesses the UNSW-NB15 dataset by performing various data cleaning and transformation steps.

Functions:
- remove_outliers(df: pd.DataFrame, cols: list, threshold=1.5) -> pd.DataFrame:
  Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.

- main():
  Main function to load, clean, preprocess, and save the UNSW-NB15 dataset.

Global Variables:
-----------------
- DATASET_DIR: Path
  The directory where the UNSW-NB15 dataset files are located.

Usage:
------
Run this module as a script to preprocess the UNSW-NB15 dataset and save the cleaned and normalized data to parquet files."""

from pathlib import Path
from functools import partial
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import FunctionTransformer, make_pipeline


DATASET_DIR = Path("./datasets/UNSW-NB15")


def remove_outliers(df: pd.DataFrame, cols: list, threshold=1.5):
    """
    Remove outliers from a DataFrame based on the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : DataFrame
    cols : list
        List of column names to consider for outlier detection.
    threshold : float, default is 1.5
        The multiplier for the IQR to define the outlier range.

    Returns
    -------
    DataFrame
        A DataFrame with outliers removed based on the specified columns and threshold.
    """
    q1, q3 = map(
        partial(pd.Series, index=cols),
        df[cols].quantile([0.25, 0.75]).to_numpy(),
    )
    iqr = q3 - q1
    return df[
        (
            (df[cols] >= (q1 - threshold * iqr)) & (df[cols] <= (q3 + threshold * iqr))
        ).all(axis=1)
    ]


def main():
    df_col: pd.DataFrame = pd.read_csv(
        DATASET_DIR / "NUSW-NB15_features.csv",
        header=0,
        names=["index", "name", "type", "description"],
        index_col="name",
        usecols=["name", "type", "description"],
        converters={
            "name": str.lower,
            "type": lambda x: {
                "binary": "boolean",
                "float": "Float64",
                "integer": "Int64",
                "nominal": "category",
                "timestamp": "datetime64[s]",
            }.get(x.lower(), "object"),
        },
        encoding="cp1252",  # :see: https://en.wikipedia.org/wiki/Windows-1252
    )

    df: pd.DataFrame = pd.concat(
        [
            pd.read_csv(
                DATASET_DIR / f"UNSW-NB15_{i}.csv",
                names=df_col.index.tolist(),
                dtype=df_col.type
                # Parse DateTime and Int64 columns later
                .replace(["datetime64[s]", "Int64"], "object")
                # Some rows have `is_ftp_login`: '2', '4'
                .mask(df_col.index == "is_ftp_login", "Int64").to_dict(),
                na_values=[" ", "-", "no"],
            )
            for i in range(1, 5)
        ],
        ignore_index=True,
    ).drop_duplicates(
        keep="first"
    )  # Early heuristic, drops 480633/2059414 duplicates

    # Parse columns with type Int64 using `int` function to infer string encoding
    # due to mixed integer, hexadecimal, and string values
    df.loc[:, df_col.type == "Int64"] = df.loc[:, df_col.type == "Int64"].map(
        int, na_action="ignore", base=0
    )

    # Parse columns with type datetime64[s] using `pd.to_datetime` function
    df.loc[:, df_col.type == "datetime64[s]"] = df.loc[
        :, df_col.type == "datetime64[s]"
    ].apply(pd.to_datetime, errors="coerce", unit="s")

    df = (
        # Drop invalid boolean values in `is_ftp_login` column
        df.drop(df.loc[df.is_ftp_login.isin([2, 4])].index)
        # Drop rows with missing values in `sport`, `dsport`` columns
        .dropna(subset=["sport", "dsport", "state"])
        # Cast columns to their data types
        .astype(df_col.type, errors="raise")
        # Drop columns with high cardinality
        .drop(columns=["ct_ftp_cmd", "ct_flw_http_mthd"])
    )

    # Remove outliers for some numerical columns
    df = remove_outliers(
        df,
        [
            "dur",  # Record total duration
            "sbytes",  # Source to destination transaction bytes
            "dbytes",  # Destination to source transaction bytes
        ],
    )
    df.attrs = df_col.description.to_dict()

    print(df.info())
    print(df.head())
    print(df.shape)

    preprocessor = make_column_transformer(
        (
            make_pipeline(
                FunctionTransformer(lambda X: X.apply(pd.to_numeric, errors="coerce")),
                SimpleImputer(),
                StandardScaler(),
            ),
            make_column_selector(dtype_exclude="category"),  # type: ignore[arg-type]
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
            make_column_selector(dtype_include="category"),  # type: ignore[arg-type]
        ),
    )
    preprocessor.set_output(transform="pandas")
    # Takes ~3m to run
    normalized_df = df.transform(preprocessor.fit_transform)  # type: ignore[arg-type]

    df.to_parquet(DATASET_DIR / "UNSW-NB15.parquet", index=True)
    normalized_df.to_parquet(DATASET_DIR / "UNSW-NB15_normalized.parquet", index=True)


if __name__ == "__main__":
    main()
