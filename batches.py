import numpy as np
from tensorflow.keras.utils import Sequence  # type: ignore
class Batches(Sequence):
    '''
    Custom batch generator for training models
    '''

    def __init__(self, x_data : np.ndarray, y_data : np.ndarray, batch_size: int = 32):

        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.class_distribution = y_data.sum(axis=0)
        self.num_zeros = np.sum(y_data == 0)
        self.num_ones = np.sum(y_data == 1)
        self.class_0_indices : np.ndarray = np.where((self.y_data == 0).any(axis=1))[0]
        self.class_1_indices : np.ndarray = np.where((self.y_data == 1).all(axis=1))[0]

        self.num_0 = len(self.class_0_indices)
        self.num_1 = len(self.class_1_indices)

        self.size = min(self.num_0,self.num_1)

    def __getitem__(self, index):

        num_class_0 : int = int(self.batch_size * 0.5)
        num_class_1 : int = int(self.batch_size * 0.5)

        class_0_indices : np.ndarray = np.where((self.y_data == 0).any(axis=1))[0]
        class_1_indices : np.ndarray = np.where((self.y_data == 1).all(axis=1))[0]

        batch_indices_0 : np.ndarray = class_0_indices[index * num_class_0 : (index + 1) * num_class_0 ]
        batch_indices_1 : np.ndarray = class_1_indices[index * num_class_1 : (index + 1) * num_class_1 ]


        batch_indices : np.ndarray = np.concatenate((batch_indices_0, batch_indices_1))
        np.random.shuffle(batch_indices)

        batch_x : np.ndarray = self.x_data[batch_indices]
        batch_y : np.ndarray = self.y_data[batch_indices]

        return batch_x, batch_y

    def __len__(self):
        return 2 * self.size // self.batch_size
