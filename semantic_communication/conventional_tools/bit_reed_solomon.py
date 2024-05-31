import numpy as np
from galois import ReedSolomon

from semantic_communication.conventional_tools.bit_operator import BitOperator


class BitReedSolomon(ReedSolomon):
    def __init__(self, n, k, m):
        self.m = m
        super().__init__(n, k)

    def encode_bit_sequence(self, bit_sequence: np.array):
        symbols = BitOperator.bits_to_symbols(bit_sequence, self.m)
        n_symbols = len(symbols)

        if n_symbols % self.k != 0:
            n_full_messages = (n_symbols // self.k) * self.k
            full_messages = symbols[:n_full_messages].reshape(-1, self.k)
            short_message = symbols[n_full_messages:]

            full_messages_encoded = self.encode(full_messages).flatten()
            short_message_encoded = self.encode(short_message).flatten()

            encoded_symbols = np.concatenate(
                [full_messages_encoded, short_message_encoded]
            )
        else:
            symbols = symbols.reshape(-1, self.k)
            encoded_symbols = self.encode(symbols).flatten()

        encoded_bits = BitOperator.symbols_to_bits(encoded_symbols, self.m)
        return encoded_bits

    def decode_bit_sequence(self, bit_sequence: np.array):
        symbols = BitOperator.bits_to_symbols(bit_sequence, self.m)
        n_symbols = len(symbols)

        if n_symbols % self.n != 0:
            n_full_messages = (n_symbols // self.n) * self.n
            full_messages = symbols[:n_full_messages].reshape(-1, self.n)
            short_message = symbols[n_full_messages:]

            full_messages_decoded = self.decode(full_messages).flatten()
            short_message_decoded = self.decode(short_message).flatten()

            decoded_symbols = np.concatenate(
                [full_messages_decoded, short_message_decoded]
            )
        else:
            symbols = symbols.reshape(-1, self.n)
            decoded_symbols = self.decode(symbols).flatten()

        decoded_bits = BitOperator.symbols_to_bits(decoded_symbols, self.m)
        return decoded_bits