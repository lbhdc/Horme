from pandas import concat, DataFrame


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
