import numpy as np


class BitOperator:
    @classmethod
    def bits_to_symbols(cls, bits: np.array, m: int) -> np.array:
        return cls.bits_to_int(bits.reshape(-1, m))

    @staticmethod
    def bits_to_int(a):
        return a.dot(1 << np.arange(a.shape[-1] - 1, -1, -1))

    @staticmethod
    def symbols_to_bits(symbols: np.array, m: int) -> np.array:
        symbols = symbols.reshape(-1, 1)
        r_shift = (1 << np.arange(m))[::-1]

        bits = ((symbols & r_shift) > 0).astype(np.uint8).flatten()
        return bits