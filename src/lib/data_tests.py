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
        data = self.sample_data(100_000)
        max_count = data.target.value_counts().max()
        min_count = data.target.value_counts().min()

        self.assertNotEqual(
            max_count, min_count,
            "sample data is already balanced"
        )

        balanced_data = data.pipe(lib.data.balance())
        results = []

        for result in balanced_data.target.value_counts():
            results.append(min_count == result)

        err = results.count(False)
        self.assertEqual(
            0, err,
            "balance did not underbalance with the default settings"
        )

    def test_oversample(self):
        data = self.sample_data(100_000)
        max_count = data.target.value_counts().max()
        min_count = data.target.value_counts().min()

        self.assertNotEqual(
            max_count, min_count,
            "sample data is already balanced"
        )

        balanced_data = data.pipe(lib.data.balance(strategy="oversample"))
        results = []

        for result in balanced_data.target.value_counts():
            results.append(max_count == result)

        err = results.count(False)
        self.assertEqual(
            0, err,
            "balance did not overbalance with the default settings"
        )