class JavaRandom:
    """java.util.Random-compatible generator.

    This is used during the Python migration to preserve deterministic behavior
    between Java and Python runs.
    """

    _MULT = 0x5DEECE66D
    _ADD = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed):
        self.seed = 0
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = (int(seed) ^ self._MULT) & self._MASK

    def next(self, bits):
        self.seed = (self.seed * self._MULT + self._ADD) & self._MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound=None):
        if bound is None:
            value = self.next(32)
            return value - (1 << 32) if value >= (1 << 31) else value

        if bound <= 0:
            raise ValueError("bound must be positive")

        if (bound & (bound - 1)) == 0:
            return (bound * self.next(31)) >> 31

        while True:
            bits = self.next(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value

    def next_long(self, bound=None):
        if bound is not None:
            bound = int(bound)
            if bound <= 0:
                raise ValueError("bound must be positive")
            m = bound - 1
            r = self.next_long()
            if (bound & m) == 0:
                return r & m
            u = (r & ((1 << 64) - 1)) >> 1
            while True:
                value = u % bound
                if u + m - value >= 0:
                    return value
                u = ((self.next_long() & ((1 << 64) - 1)) >> 1)

        upper = self.next(32)
        lower = self.next(32)
        if upper >= (1 << 31):
            upper -= (1 << 32)
        if lower >= (1 << 31):
            lower -= (1 << 32)
        value = (upper << 32) + lower
        # Keep value in signed 64-bit range to mirror Java long overflow.
        value = ((value + (1 << 63)) % (1 << 64)) - (1 << 63)
        return value

    def next_boolean(self):
        return self.next(1) != 0

    def next_float(self):
        return self.next(24) / float(1 << 24)

    def next_double(self):
        high = self.next(26)
        low = self.next(27)
        return ((high << 27) + low) / float(1 << 53)
