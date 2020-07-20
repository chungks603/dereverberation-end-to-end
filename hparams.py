import os
import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Union, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class HParams(object):
    """
    If you don't understand 'field(init=False)' and __post_init__,
    read python 3.7 dataclass documentation
    """
    # Dataset Settings
    feature: str = 'IV'
    path_feature: Path = Path('/mnt/sdb1/chungks/dereverberation/data')
    room_train: str = 'fitting'
    room_test: str = 'fitting'

    # Feature Parameters
    sample_rate: int = 16000
    refresh_normconst: bool = False

    # STFT Parameters
    kwargs_stft = dict(n_fft=512, hop_length=256, win_length=512)
    kwargs_istft = dict(hop_length=256, win_length=512)

    # Drawing Spectrogram Parameters
    kwargs_spec = dict(vmin=-50, vmax=20)

    # Summary path
    logdir: str = f'./runs/'

    # Model Parameters
    model: Dict[str, Any] = field(init=False)

    # Training Parameters
    scheduler: Dict[str, Any] = field(init=False)
    train_ratio: float = 0.87
    batch_size: int = 16
    num_epochs: int = 300
    learning_rate: float = 5e-4
    weight_decay: float = 0.

    # Device-dependent Parameters
    # 'cpu', 'cuda:n', the cuda device no., or the tuple of the cuda device no.
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 3
    num_workers: int = 4

    def __post_init__(self):
        self.model = dict(n_features=4,
                          n_layers=9,
                          ch_interval=36,
                          ch_double=False)

        self.scheduler = dict(restart_period=2,
                              t_mult=2,
                              eta_threshold=1.5,)

        self.dummy_input: Tuple = (self.model['n_features'], 2 ** (self.model['n_layers'] + 2))
        self.l_pad = 2 ** (self.model['n_layers'] - 1)

        # path
        form = f'{self.feature}_{{}}'
        p_f_train = self.path_feature / f'{form.format(self.room_train)}/TRAIN_8_1200'
        p_f_valid = self.path_feature / f'{form.format(self.room_train)}/VALID_8'
        p_f_test = self.path_feature / f'{form.format(self.room_test)}/TEST_8'
        self.dict_path = dict(
            feature_train=p_f_train,
            feature_valid=p_f_valid,
            feature_test=p_f_test,

            normconst_train=p_f_train / 'normconst.npz',
        )

    @staticmethod
    def is_featurefile(f: os.DirEntry) -> bool:
        return (f.name.endswith('.npz')
                and not f.name.startswith('metadata')
                and not f.name.startswith('normconst'))

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True):
        if not parser:
            parser = argparse.ArgumentParser()
        dict_self = asdict(self)
        for k in dict_self:
            parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            parsed = getattr(args, k)
            if parsed == '':
                continue
            if type(dict_self[k]) == str:
                setattr(self, k, parsed)
            else:
                v = eval(parsed)
                if isinstance(v, type(dict_self[k])):
                    setattr(self, k, eval(parsed))

        if print_argument:
            self.print_params()

        return args

    def print_params(self):
        print('-------------------------')
        print('Hyper Parameter Settings')
        print('-------------------------')
        for k, v in asdict(self).items():
            print(f'{k}: {v}')
        print('-------------------------')


hp = HParams()
