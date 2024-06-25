

import numpy as np

class BinaryArrayCompressor:
    def __init__(self, array=None):
        if array is not None:
            self.original_shape = array.shape
            self.compressed_data = self.compress(array)
        else:
            self.original_shape = None
            self.compressed_data = None

    def compress(self, array):
        # 确保输入是一个只包含0和1的numpy数组
        assert isinstance(array, np.ndarray), "Input should be a numpy array."
        assert np.array_equal(array, array.astype(bool)), "Array should only contain 0 and 1."

        # 将数组展平并计算需要多少个字节来存储这些位
        flat_array = array.flatten()
        n_bits = flat_array.size
        n_bytes = (n_bits + 7) // 8  # 向上取整

        # 创建一个空的字节数组
        compressed_data = np.zeros(n_bytes, dtype=np.uint8)

        # 将每个bit写入字节数组中
        for i in range(n_bits):
            byte_index = i // 8
            bit_index = i % 8
            if flat_array[i] > 0:
                compressed_data[byte_index] |= (1 << bit_index)

        return compressed_data

    def decompress(self, v=1):
        if self.compressed_data is None or self.original_shape is None:
            raise ValueError("No data to decompress.")

        n_bits = np.prod(self.original_shape)
        flat_array = np.zeros(n_bits, dtype=np.uint8)

        for i in range(n_bits):
            byte_index = i // 8
            bit_index = i % 8
            flat_array[i] = (self.compressed_data[byte_index] >> bit_index) & 1

        return flat_array.reshape(self.original_shape) * v

    def save(self, filename):
        np.savez_compressed(filename, compressed_data=self.compressed_data, original_shape=self.original_shape)

    def load(self, filename):
        with np.load(filename) as data:
            self.compressed_data = data['compressed_data']
            self.original_shape = tuple(data['original_shape'])

