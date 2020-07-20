import contextlib
import os
from pathlib import Path
from typing import Callable, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def draw_spec(data: tuple, fs: int, **kwargs):
    """
    :return: magnitude spectrogram of clean, noisy, enhanced speech
    """
    wav_type = ['Clean', 'Noisy', 'Enhanced']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, dpi=150)
    for ax, dat, title in zip((ax1, ax2, ax3), data, wav_type):
        ax.imshow(dat,
                  cmap=plt.get_cmap('CMRmap'),
                  extent=(0, dat.shape[1], 0, fs // 2),
                  origin='lower', aspect='auto', **kwargs)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        fig.colorbar(ax.images[0], format='%+2.0f dB')
    fig.tight_layout()

    return fig


def print_eval(eval_xs: Dict, eval_outs: Dict):
    """
    print evaluation results of speech quality metrics
    :param eval_outs:
    :param eval_xs:
    :return:
    """
    print(f'Test - Input Eval: {eval_xs}')
    print(f'Test - Out Eval: {eval_outs}')


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)


def pad_sequence(sequences, batch_first=False, padding_value=0, len_factor=1):
    """
    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        len_factor (int, optional): value for a factor of the length to be. Default: 1.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])

    rem = max_len % len_factor
    if rem != 0:
        max_len += len_factor - rem

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def get_lr(optim):
    return optim.param_groups[0]["lr"]


def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr


def set_cyclic_lr(optimizer, i_batch, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(i_batch // cycle_length, cycles - 1)
    curr_it = i_batch - cycle_length * curr_cycle

    new_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)


def convert(a: Union[Tensor, ndarray], astype: type, device=None):
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, dtype=torch.float32, device=device)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)


class DataPerDevice:
    """
    Converting ndarray to tensors of cpu or cuda devices when it is firstly needed,
    and keep the tensors for the next usage.
    """
    __slots__ = ('data',)

    def __init__(self, data_np: ndarray):
        self.data = {ndarray: data_np}

    def __getitem__(self, typeOrtup):
        if type(typeOrtup) == tuple:
            _type, device = typeOrtup
        elif typeOrtup == ndarray:
            _type = ndarray
            device = None
        else:
            raise IndexError

        if _type == ndarray:
            return self.data[ndarray]
        else:
            if typeOrtup not in self.data:
                self.data[typeOrtup] = convert(self.data[ndarray].astype(np.float32),
                                               Tensor,
                                               device=device)
            return self.data[typeOrtup]

    def get_like(self, other):
        if type(other) == Tensor:
            return self[Tensor, other.device]
        else:
            return self[ndarray]
