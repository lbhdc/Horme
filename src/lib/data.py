import os
from pandas import Categorical, concat, DataFrame, read_csv


def balance(group="target", strategy="undersample", seed=0):
    def F(df):
        count: int
        replace: bool
        target = DataFrame()

        # oversample
        if "oversample" in strategy:
            count = df[group].value_counts().max()
            replace = True

        # undersample
        else:
            count = df[group].value_counts().min()
            replace = False

        for action in df[group].unique():
            sample = (
                df[df[group] == action]
                    .sample(n=count, replace=replace, random_state=seed)
            )

            target = concat((target, sample), axis="rows")

        return target

    return F


def read(path):
    return (
        read_csv(path)
        .rename(columns=str.lower)
        .rename(columns={"gt": "target"})
        .set_index("arrival_time")
        .drop(columns=["creation_time", "device", "index", "model"])
        .assign(
            target=lambda df: Categorical(df["target"]),
            user=lambda df: Categorical(df["user"])
        )
    )


def read_local_phones():
    data_directory = (
        os.environ["DATASET"] + "/heterogeneity_activity_recognition"
    )

    data_path = f"{data_directory}/processed/phones.zip"
    return read_csv(data_path)