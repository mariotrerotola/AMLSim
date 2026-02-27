import unittest

from amlsim.java_random import JavaRandom
from amlsim.models import TargetedTransactionAmount


class _StubSimProperties:
    def __init__(self, min_amount, max_amount):
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)

    def get_min_transaction_amount(self):
        return self.min_amount

    def get_max_transaction_amount(self):
        return self.max_amount


class TargetedAmountTests(unittest.TestCase):

    def test_static_sampler_matches_legacy_object_path(self):
        sim_properties = _StubSimProperties(100.0, 1000.0)
        random_a = JavaRandom(42)
        random_b = JavaRandom(42)

        legacy_value = TargetedTransactionAmount(500.0, sim_properties, random_a).double_value()
        static_value = TargetedTransactionAmount.sample(500.0, 100.0, 1000.0, random_b)

        self.assertEqual(legacy_value, static_value)

    def test_small_gap_keeps_target_amount(self):
        random_source = JavaRandom(7)
        sampled = TargetedTransactionAmount.sample(150.0, 100.0, 1000.0, random_source)
        self.assertEqual(sampled, 150.0)


if __name__ == "__main__":
    unittest.main()
