import lib.data
import pandas as pd
import random
import string
import unittest


class BalanceTests(unittest.TestCase):

    def sample_data(self, n):
        data = [random.choice(string.ascii_letters) for _ in range(n)]
        return pd.DataFrame({"target": data}, dtype="category")

    def test_undersample(self):
        # a random column name is generated to make sure a schema is being
        # relied unknowingly.
        col = "".join([random.choice(string.ascii_letters) for _ in range(5)])

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
            "balance did not underbalance with the default settings"
        )

    def test_oversample(self):
        # a random column name is generated to make sure a schema is being
        # relied unknowingly.
        col = "".join([random.choice(string.ascii_letters) for _ in range(5)])

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
            "balance did not overbalance with the default settings"
        )