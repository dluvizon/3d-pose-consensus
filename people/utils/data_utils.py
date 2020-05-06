import numpy as np

import random
from multiprocessing import Queue
import queue
import threading

import keras.backend as K

from keras.utils import Sequence

# Data definitions
TEST_MODE = 0
TRAIN_MODE = 1
VALID_MODE = 2


def get_clip_frame_index(sequence_size, subsample, num_frames,
        random_clip=False):

    # Assert that subsample is integer and positive
    assert (type(subsample) == int) and subsample > 0

    idx_coef = 1.
    while idx_coef*sequence_size < num_frames:
        idx_coef *= 1.5
    sequence_size *= idx_coef

    # Check if the given subsample value is feasible, otherwise, reduce
    # it to the maximum acceptable value.
    max_subsample = int(sequence_size / num_frames)
    if subsample > max_subsample:
        subsample = max_subsample

    vidminf = subsample * (num_frames - 1) + 1 # Video min num of frames
    maxs = sequence_size - vidminf # Maximum start
    if random_clip:
        start = np.random.randint(maxs + 1)
    else:
        start = int(maxs / 2)

    frames = list(range(start, start + vidminf, subsample))
    if idx_coef > 1:
        for i in range(len(frames)):
            frames[i] = int(frames[i] / idx_coef)

    return frames


def calc_number_of_poses(img_shape, anchors):
    num_poses = 0
    for anc in anchors:
        anc_poses = 1
        for d in range(2):
            assert img_shape[d] >= anc[d], \
                    'Invalid anchor {} for image shape {}'.format(anc,
                            img_shape)
            anc_poses *= int(img_shape[d] - anc[d]) + 1
        num_poses += anc_poses

    return num_poses


class DataConfig(object):
    """Input frame configuration and data augmentation setup."""

    def __init__(self, crop_resolution=(256, 256), image_channels=(3,),
            angles=[0], fixed_angle=0,
            scales=[1], fixed_scale=1,
            trans_x=[0], fixed_trans_x=0,
            trans_y=[0], fixed_trans_y=0,
            hflips=[0, 1], fixed_hflip=0,
            chpower=0.01*np.array(range(80, 120+1, 10)), fixed_chpower=1,
            geoocclusion=None, fixed_geoocclusion=None,
            subsampling=[1], fixed_subsampling=1):

        self.crop_resolution = crop_resolution
        self.image_channels = image_channels
        if K.image_data_format() == 'channels_last':
            self.input_shape = crop_resolution[-1::-1] + image_channels
        else:
            self.input_shape = image_channels + crop_resolution[-1::-1]
        self.angles = angles
        self.fixed_angle = fixed_angle
        self.scales = scales
        self.fixed_scale = fixed_scale
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.fixed_trans_x = fixed_trans_x
        self.fixed_trans_y = fixed_trans_y
        self.hflips = hflips
        self.fixed_hflip = fixed_hflip
        self.chpower = chpower
        self.fixed_chpower = fixed_chpower
        self.geoocclusion = geoocclusion
        self.fixed_geoocclusion = fixed_geoocclusion
        self.subsampling = subsampling
        self.fixed_subsampling = fixed_subsampling

    def get_fixed_config(self):
        return {'angle': self.fixed_angle,
                'scale': self.fixed_scale,
                'transx': self.fixed_trans_x,
                'transy': self.fixed_trans_y,
                'hflip': self.fixed_hflip,
                'chpower': self.fixed_chpower,
                'geoocclusion': self.fixed_geoocclusion,
                'subspl': self.fixed_subsampling}

    def random_data_generator(self):
        angle = DataConfig._getrand(self.angles)
        scale = DataConfig._getrand(self.scales)
        trans_x = DataConfig._getrand(self.trans_x)
        trans_y = DataConfig._getrand(self.trans_y)
        hflip = DataConfig._getrand(self.hflips)
        chpower = (DataConfig._getrand(self.chpower),
                DataConfig._getrand(self.chpower),
                DataConfig._getrand(self.chpower))
        geoocclusion = self.__get_random_geoocclusion()
        subsampling = DataConfig._getrand(self.subsampling)

        return {'angle': angle,
                'scale': scale,
                'transx': trans_x,
                'transy': trans_y,
                'hflip': hflip,
                'chpower': chpower,
                'geoocclusion': geoocclusion,
                'subspl': subsampling}

    def __get_random_geoocclusion(self):
        if self.geoocclusion is not None:

            w = int(DataConfig._getrand(self.geoocclusion) / 2)
            h = int(DataConfig._getrand(self.geoocclusion) / 2)
            xmin = w + 1
            xmax = self.crop_resolution[0] - xmin
            ymin = h + 1
            ymax = self.crop_resolution[1] - ymin

            x = DataConfig._getrand(range(xmin, xmax, 5))
            y = DataConfig._getrand(range(ymin, ymax, 5))
            bbox = (x-w, y-h, x+w, y+h)

            return bbox

        else:
            return None

    @staticmethod
    def _getrand(x):
        return x[np.random.randint(0, len(x))]


class BatchLoader(Sequence):
    """Loader class for generic datasets, based on the Sequence class from
    Keras.

    One (or more) object(s) implementing a dataset should be provided.
    The required functions are 'get_length(self, mode)' and
    'get_data(self, key, mode)'. The first returns an integer, and the last
    returns a dictionary containing the data for a given pair of (key, mode).

    # Arguments
        dataset: A dataset object, or a list of dataset objects (for multiple
            datasets), which are merged by this class.
        x_dictkeys: Key names (strings) to constitute the baches of X data
            (input).
        y_dictkeys: Identical to x_dictkeys, but for Y data (labels).
            All given datasets must provide those keys.
        batch_size: Number of samples in each batch. If multiple datasets, it
            can be a list with the same length of 'dataset', where each value
            corresponds to the number of samples from the respective dataset,
            or it can be a single value, which corresponds to the number of
            samples from *each* dataset.
        num_predictions: number of predictions (y) that should be repeated for
            training.
        mode: TRAIN_MODE, TEST_MODE, or VALID_MODE.
        shuffle: boolean to shuffle *samples* (not batches!) or not.
    """
    BATCH_HOLD = 8

    def __init__(self, dataset, x_dictkeys, y_dictkeys, mode,
            batch_size=24, num_predictions=1, interlaced=False, shuffle=True):

        if not isinstance(dataset, list):
            dataset = [dataset]
        self.datasets = dataset
        self.x_dictkeys = x_dictkeys
        self.y_dictkeys = y_dictkeys
        self.allkeys = x_dictkeys + y_dictkeys

        """Make sure that all datasets have the same shapes for all dictkeys"""
        for dkey in self.allkeys:
            for i in range(1, len(self.datasets)):
                assert self.datasets[i].get_shape(dkey) == \
                        self.datasets[i-1].get_shape(dkey), \
                        'Incompatible dataset shape for dictkey {}'.format(dkey)

        self.batch_sizes = batch_size
        if not isinstance(self.batch_sizes, list):
            self.batch_sizes = len(self.datasets)*[self.batch_sizes]

        assert len(self.datasets) == len(self.batch_sizes), \
                'dataset and batch_size should be lists with the same length.'

        if isinstance(num_predictions, int):
            self.num_predictions = len(self.y_dictkeys)*[num_predictions]
        elif isinstance(num_predictions, list):
            if interlaced:
                raise ValueError('List of `num_predictions` not supported '
                        'with `interlaced` option!')
            self.num_predictions = num_predictions
        else:
            raise ValueError(
                'Invalid num_predictions ({})'.format(num_predictions))

        self.interlaced = interlaced

        assert len(self.num_predictions) == len(self.y_dictkeys), \
                'num_predictions and y_dictkeys not matching'

        self.mode = mode
        self.shuffle = shuffle

        """Create one lock object for each dataset in case of data shuffle."""
        if self.shuffle:
            self.qkey = []
            self.lock = []
            for d in range(self.num_datasets):
                maxsize = self.datasets[d].get_length(self.mode) \
                        + BatchLoader.BATCH_HOLD*self.batch_sizes[d]
                self.qkey.append(Queue(maxsize=maxsize))
                self.lock.append(threading.Lock())

    def __len__(self):
        dataset_len = []
        for d in range(self.num_datasets):
            dataset_len.append(
                    int(np.ceil(self.datasets[d].get_length(self.mode) /
                        float(self.batch_sizes[d]))))

        return max(dataset_len)


    def __getitem__(self, idx):
        data_dict = self.get_data(idx, self.mode)

        """Convert the dictionary of samples to a list for x and y."""
        x_batch = []
        for dkey in self.x_dictkeys:
            x_batch.append(data_dict[dkey])

        y_batch = []
        if self.interlaced:
            for _ in range(self.num_predictions[0]):
                for i, dkey in enumerate(self.y_dictkeys):
                    y_batch.append(data_dict[dkey])
        else:
            for i, dkey in enumerate(self.y_dictkeys):
                for _ in range(self.num_predictions[i]):
                    y_batch.append(data_dict[dkey])

        return x_batch, y_batch

    def get_batch_size(self):
        return sum(self.batch_sizes)

    def get_data(self, idx, mode):
        """Get the required data by mergning all the datasets as specified
        by the object's parameters."""
        data_dict = {}
        for dkey in self.allkeys:
            data_dict[dkey] = np.empty((sum(self.batch_sizes),) \
                    + self.datasets[0].get_shape(dkey))

        batch_cnt = 0
        for d in range(len(self.datasets)):
            for i in range(self.batch_sizes[d]):
                if self.shuffle:
                    key = self.get_shuffled_key(d)
                else:
                    key = idx*self.batch_sizes[d] + i
                    if key >= self.datasets[d].get_length(mode):
                        key -= self.datasets[d].get_length(mode)

                data = self.datasets[d].get_data(key, mode)
                for dkey in self.allkeys:
                    data_dict[dkey][batch_cnt, :] = data[dkey]

                batch_cnt += 1

        return data_dict

    def get_shape(self, dictkey):
        """Inception of get_shape method.
        """
        return (sum(self.batch_sizes),) + self.datasets[0].get_shape(dictkey)

    def get_length(self, mode):
        assert mode == self.mode, \
                'You are mixturing modes! {} with {}'.format(mode, self.mode)
        return len(self)

    def get_shuffled_key(self, dataset_idx):
        assert self.shuffle, \
                'There is not sense in calling this function if shuffle=False!'

        key = None
        with self.lock[dataset_idx]:
            min_samples = BatchLoader.BATCH_HOLD*self.batch_sizes[dataset_idx]
            if self.qkey[dataset_idx].qsize() <= min_samples:
                """Need to fill that list."""
                num_samples = self.datasets[dataset_idx].get_length(self.mode)
                newlist = list(range(num_samples))
                random.shuffle(newlist)
                try:
                    for j in newlist:
                        self.qkey[dataset_idx].put(j, False)
                except queue.Full:
                    pass
            key = self.qkey[dataset_idx].get()

        return key

    @property
    def num_datasets(self):
        return len(self.datasets)


class ConcatBatchLoader(Sequence):

    def __init__(self, batch_loaders):
        assert isinstance(batch_loaders, list),\
                'batch_loaders should be a list of BatchLoader objects'
        self.batch_loaders = batch_loaders

    def __len__(self):
        return min([len(b) for b in self.batch_loaders])

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []

        for loader in self.batch_loaders:
            x, y = loader[idx]
            for k in x:
                x_batch.append(k)
            for k in y:
                y_batch.append(k)

        return x_batch, y_batch


