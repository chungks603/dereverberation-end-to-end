"""
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
"""
import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import scipy.io as scio
import torch
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from emph import pre_emphasis
from utils import DataPerDevice, pad_sequence
from hparams import hp


class Normalization:
    """
    Calculating and saving std of all data with respect to time axis,
    applying normalization.
    It only includes standard deviation division/multiplication for
    time domain features, which have zero-mean.
    This is needed only when you don't load all the data on the RAM
    """

    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum(axis=0, keepdims=True)

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a) ** 2).sum(axis=0, keepdims=True)

    @staticmethod
    def _load_data(fname: Union[str, Path], key: str, queue: mp.Queue) -> None:
        x = np.load(fname)[key].astype(np.float32)
        queue.put(x)

    @staticmethod
    def _calc_per_data(data,
                       list_func: Sequence[Callable],
                       args: Sequence = None,
                       ) -> Dict[Callable, Any]:
        """ gather return values of functions in `list_func`

        :param list_func:
        :param args:
        :return:
        """

        if args:
            result = {f: f(data, arg) for f, arg in zip(list_func, args)}
        else:
            result = {f: f(data) for f in list_func}
        return result

    def __init__(self, std):
        # self.mean = DataPerDevice(mean.astype(np.float32, copy=False))
        self.std = DataPerDevice(std.astype(np.float32, copy=False))

    @classmethod
    def calc_const(cls, all_files: List[Path], key: str):
        """

        :param all_files:
        :param key: data name in npz file
        :rtype: Normalization
        """

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum)
        pool_loader = mp.Pool(4)
        pool_calc = mp.Pool(min(mp.cpu_count() - 2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                ))

        result: List[Dict] = [item.get() for item in result]

        sum_size = np.sum([item[np.size] for item in result])
        sum_ = np.sum([item[cls._sum] for item in result],
                      axis=0, dtype=np.float32)
        mean = sum_ / (sum_size // sum_.size)

        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                ))

        pool_loader.close()
        pool_calc.close()
        result: List[Dict] = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result],
                            axis=0, dtype=np.float32)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        # Conserve the directional information (X, Y, Z)
        if key == hp.feature:
            std[-1, 1:] = np.sqrt(np.sum(std[-1, 1:] ** 2))
        print('Calculated Std')

        return cls(std)

    def astuple(self):
        return self.std.data[ndarray]

    # normalize and denormalize functions can accept a ndarray or a tensor.
    def normalize(self, a):
        return a / self.std.get_like(a)

    def normalize_(self, a):  # in-place version
        a /= self.std.get_like(a)

        return a

    def denormalize(self, a):
        return a * self.std.get_like(a)

    def denormalize_(self, a):  # in-place version
        a *= self.std.get_like(a)

        return a


class CustomDataset(Dataset):
    def __init__(self, kind_data: str,
                 norm_modules: Dict[str, Normalization] = None):
        self._PATH: Path = hp.dict_path[f'feature_{kind_data}']

        # TODO: file list
        self._all_files = [f for f in self._PATH.glob('*.*') if hp.is_featurefile(f)]
        self._all_files.sort()
        # if all files can be loaded on RAM, load all the data

        self.norm_modules = dict()
        if kind_data == 'train':
            path_normconst = hp.dict_path[f'normconst_{kind_data}']

            if path_normconst.exists() and not hp.refresh_normconst:
                npz_normconst = np.load(path_normconst, allow_pickle=True)
                self.norm_modules['x'] = Normalization(*npz_normconst['normconst_x'])
                self.norm_modules['y'] = Normalization(*npz_normconst['normconst_y'])
            else:
                self.norm_modules['x'] = Normalization.calc_const(self._all_files, key=hp.feature)
                self.norm_modules['y'] = Normalization.calc_const(self._all_files, key='wav')
                np.savez(path_normconst,
                         normconst_x=self.norm_modules['x'].astuple(),
                         normconst_y=self.norm_modules['y'].astuple())
                scio.savemat(path_normconst.with_suffix('.mat'),
                             dict(normconst_x=self.norm_modules['x'].astuple(),
                                  normconst_y=self.norm_modules['y'].astuple()))

                print(f'normalization consts for input: {self.norm_modules["x"]}')
                print(f'normalization consts for output: {self.norm_modules["y"]}')
        else:
            assert 'x' in norm_modules and 'y' in norm_modules
            self.norm_modules = norm_modules

    def __getitem__(self, idx: int) -> Dict:
        """
        :param idx:
        :return:
        """
        pair = np.load(self._all_files[idx])

        # feature
        x = pair[hp.feature]
        x = pre_emphasis(x, axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # clean speech
        y = pair['wav']
        y = pre_emphasis(y)
        y = torch.tensor(y, dtype=torch.float32)

        sample = dict(
            x=x,
            y=y,
            T_xs=x.shape[0],
            T_ys=y.shape[0],
            fname=self._all_files[idx].name,
        )

        return sample

    def __len__(self):
        return len(self._all_files)

    @torch.no_grad()
    def custom_collate(self, batch: List[Dict]) -> Dict:
        """

        :param batch:
        :return: Dict
            batch_x : reverberant speech features with zero padding
            batch_y: clean speeches with zero padding
            Ts: time duration of clean speech
        """
        # stack samples with zero-padding
        data = dict()
        T_xs = np.array([item.pop('T_xs') for item in batch])
        T_ys = [item.pop('T_ys') for item in batch]

        data['T_xs'] = torch.tensor(T_xs, dtype=torch.int32)
        data['T_ys'] = torch.tensor(T_ys, dtype=torch.int32)

        list_fname = [item['fname'] for item in batch]
        data['fname'] = list_fname

        batch_x = [item['x'] for item in batch]
        batch_x = pad_sequence(batch_x, batch_first=True, len_factor=hp.l_pad)  # B, T, C
        batch_x = self.norm_modules['x'].normalize_(batch_x)
        batch_x = batch_x.permute(0, 2, 1)  # B, C, T

        batch_y = [item['y'] for item in batch]
        batch_y = pad_sequence(batch_y, batch_first=True, len_factor=hp.l_pad)  # B, T
        batch_y = self.norm_modules['y'].normalize_(batch_y)

        data['x'], data['y'] = batch_x, batch_y

        return data

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automatically set to the value so that the sum of the elements is 1

        :type dataset: CustomDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[Dataset]
        """
        if not isinstance(dataset, cls):
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[mask] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[mask] = 1 - ratio.sum()

        _all_files = dataset._all_files
        metadata = scio.loadmat(str(dataset._PATH / 'metadata.mat'),
                                squeeze_me=True,
                                chars_as_strings=True)
        array_n_loc = metadata['n_loc']
        if 'rooms' in metadata:
            rooms = [r.rstrip() for r in metadata['rooms']]
        else:
            array_n_loc = (array_n_loc,)
            rooms = (_all_files[0].stem.split('_')[2],)

        boundary_i_locs = np.cumsum(
            np.outer(array_n_loc, np.insert(ratio, 0, 0)),
            axis=1, dtype=np.int
        )  # number of rooms x (number of sets + 1)
        boundary_i_locs[:, -1] = array_n_loc
        i_set_per_room_loc: Dict[str, ndarray] = dict()

        for i_room, room in enumerate(rooms):
            i_set = np.empty(array_n_loc[i_room], dtype=np.int)
            for i_b in range(boundary_i_locs.shape[1] - 1):
                range_ = np.arange(boundary_i_locs[i_room, i_b],
                                   boundary_i_locs[i_room, i_b + 1])
                i_set[range_] = i_b
            i_set_per_room_loc[room] = i_set

        dataset._all_files = None
        result = [copy(dataset) for _ in range(n_split)]
        for set_ in result:
            set_._all_files = []
            # set_._needs = copy(set_._needs)
        for f in _all_files:
            _, _, room, i_loc = f.stem.split('_')
            i_loc = int(i_loc)
            result[i_set_per_room_loc[room][i_loc]]._all_files.append(f)

        return result


# Function to load numpy data and normalize, it returns dataloader for train, valid, test
def get_dataloader(hp, only_test=False):
    loader_temp = CustomDataset('train')
    loader_kwargs = dict(batch_size=hp.batch_size,
                         drop_last=False,
                         num_workers=hp.num_workers,
                         pin_memory=True,
                         collate_fn=loader_temp.custom_collate,  # if needed
                         )
    if only_test:
        speech = CustomDataset('train')  # load normalization consts
        train_loader = None
        valid_loader = None
    else:
        # create train / valid loaders
        speech = CustomDataset('train')
        train_set = speech
        valid_set = CustomDataset('valid', norm_modules=speech.norm_modules)
        # train_set, valid_set = CustomDataset.split(speech, (hp.train_ratio, -1))
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

    # test loader
    test_set = CustomDataset('test',
                             norm_modules=speech.norm_modules,
                             )
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader
