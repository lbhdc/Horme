import lib.data
import pandas as pd
import random
import string
import unittest


class BalanceTests(unittest.TestCase):

    # random target names are generated to ensure magic strings aren't being
    # relied on
    def random_target_name(self, n=5):
        return "".join([random.choice(string.ascii_letters) for _ in range(n)])

    def sample_data(self, n):
        data = [random.choice(string.ascii_letters) for _ in range(n)]
        return pd.DataFrame({"target": data}, dtype="category")

    def test_undersample(self):
        col = self.random_target_name()

        data = self.sample_data(1_000)
        data = data.rename(columns={"target": col})

        max_count = data[col].value_counts().max()
        min_count = data[col].value_counts().min()

        self.assertNotEqual(
            max_count, min_count,
            "sample data is already balanced"
        )

        balanced_data = data.pipe(lib.data.balance(group=col))
        results = []

        for result in balanced_data[col].value_counts():
            results.append(min_count == result)

        err = results.count(False)
        self.assertEqual(
            0, err,
            "balance did not underbalance the sample data"
        )

    def test_oversample(self):
        col = self.random_target_name()

        data = self.sample_data(1_000)
        data = data.rename(columns={"target": col})

        max_count = data[col].value_counts().max()
        min_count = data[col].value_counts().min()

        self.assertNotEqual(
            max_count, min_count,
            "sample data is already balanced"
        )

        balanced_data = data.pipe(
            lib.data.balance(group=col, strategy="oversample")
        )

        results = []

        for result in balanced_data[col].value_counts():
            results.append(max_count == result)

        err = results.count(False)
        self.assertEqual(
            0, err,
            "balance did not overbalance the sample data"
        )