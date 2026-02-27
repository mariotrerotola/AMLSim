import unittest

from amlsim.entities import Account


class _SequenceRandom:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0

    def next_int(self, bound=None):
        if bound is None:
            bound = 1
        value = self.sequence[self.index]
        self.index += 1
        return value % bound


class AccountEntityTests(unittest.TestCase):

    def test_tx_type_cache_is_invalidated_on_update(self):
        random_source = _SequenceRandom([0, 0])
        account = Account("src", 100.0, "b1", random_source)
        bene_a = Account("a", 0.0, "b1", random_source)
        bene_b = Account("b", 0.0, "b1", random_source)
        unknown = Account("x", 0.0, "b1", random_source)

        account.add_tx_type(bene_a, "WIRE")
        account.add_tx_type(bene_b, "ACH")
        self.assertEqual(account.get_tx_type(unknown), "WIRE")

        account.add_tx_type(bene_a, "CARD")
        self.assertEqual(account.get_tx_type(unknown), "CARD")


if __name__ == "__main__":
    unittest.main()
