import unittest

from amlsim.java_random import JavaRandom


class JavaRandomTests(unittest.TestCase):

    def test_seed_0_matches_java_reference(self):
        random = JavaRandom(0)
        self.assertEqual(random.next_int(), -1155484576)
        self.assertEqual(random.next_int(10), 8)
        self.assertEqual(random.next_long(), 4437113781045784766)
        self.assertTrue(random.next_boolean())
        self.assertAlmostEqual(random.next_float(), 0.30905056, places=8)
        self.assertAlmostEqual(random.next_double(), 0.5504370051176339, places=16)

    def test_seed_123_matches_java_reference(self):
        random = JavaRandom(123)
        self.assertEqual(random.next_int(), -1188957731)
        self.assertEqual(random.next_int(1000), 450)
        self.assertEqual(random.next_long(), -167885730524958550)
        self.assertFalse(random.next_boolean())
        self.assertAlmostEqual(random.next_float(), 0.57412946, places=8)
        self.assertAlmostEqual(random.next_double(), 0.6088003703785169, places=16)

    def test_same_seed_produces_same_sequence(self):
        a = JavaRandom(42)
        b = JavaRandom(42)
        values_a = [a.next_int(1000000) for _ in range(100)]
        values_b = [b.next_int(1000000) for _ in range(100)]
        self.assertEqual(values_a, values_b)


if __name__ == ' main ':
    unittest.main()
