"""dataloaders."""
import struct
from array import array

import numpy as np


class MnistDataloader:
    """MNIST dataloader.
    
    Referece: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
    """

    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """read images labels from file."""
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        """load dataset from files"""
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath,
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath,
            self.test_labels_filepath,
        )
        return (x_train, y_train), (x_test, y_test)


def batch_generator(x_data, y_data, batch_size, shuffle=True):
    """
    生成 mini-batch 数据的生成器。
    
    :param data: numpy ndarray, 数据集
    :param batch_size: int, 每个 batch 的样本数
    :param shuffle: bool, 是否在每次迭代前打乱数据
    :return: 生成器，每次迭代返回一个 batch 的数据
    """
    if len(x_data) != len(y_data):
        raise ValueError
    data_length = len(x_data)
    indices = np.arange(data_length)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, data_length, batch_size):
        end_idx = min(start_idx + batch_size, data_length)
        idx = indices[start_idx:end_idx]

        yield x_data[idx], y_data[idx]
