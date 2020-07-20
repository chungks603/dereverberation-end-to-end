from collections import OrderedDict as ODict
from typing import IO, Sequence, Tuple, Union, List, Dict
from itertools import repeat

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
from numpy import ndarray
import scipy.signal as scsig

from hparams import hp
import generic as gen
from matlab_lib import Evaluation as EvalModule

EVAL_METRICS = EvalModule.metrics


def calc_snrseg_time(clean: Union[List, ndarray], est: Union[List, ndarray],
                     l_frame: int, l_hop: int, T_ys: Sequence[int] = None) \
        -> float:
    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if isinstance(clean, ndarray) and clean.ndim == 1:
        clean = clean[np.newaxis, ...]
        est = est[np.newaxis, ...]
    if T_ys is None:
        T_ys = (clean.shape[-1],)

    win = scsig.windows.hann(l_frame, False)[:, np.newaxis]

    sum_result = 0.
    for T, item_clean, item_est in zip(T_ys, clean, est):
        l_pad = l_frame - (T - l_frame) % l_hop
        item_clean = np.pad(item_clean[:T], (0, l_pad), 'constant')
        item_est = np.pad(item_est[:T], (0, l_pad), 'constant')
        clean_frames = librosa.util.frame(item_clean, l_frame, l_hop) * win
        est_frames = librosa.util.frame(item_est, l_frame, l_hop) * win

        # T
        norm_clean = np.linalg.norm(clean_frames, ord=2, axis=0)
        norm_err = (np.linalg.norm(est_frames - clean_frames, ord=2, axis=0)
                    + np.finfo(np.float32).eps)

        snrseg = np.log10(norm_clean / norm_err + np.finfo(np.float32).eps)
        np.minimum(snrseg, _LIM_UPPER, out=snrseg)
        np.maximum(snrseg, _LIM_LOWER, out=snrseg)
        sum_result += snrseg.mean()
    sum_result *= 10

    return sum_result


def calc_using_eval_module(y_clean: Union[List, ndarray], y_est: Union[List, ndarray],
                           T_ys: Sequence[int] = (0,)) -> ODict:
    """ calculate metric using EvalModule. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    if isinstance(y_clean, ndarray) and y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    keys = None
    sum_result = None
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        # noinspection PyArgumentList,PyTypeChecker
        temp: ODict = EvalModule(item_clean[:T], item_est[:T], hp.sample_rate)
        result = np.array(list(temp.values()))
        if not keys:
            keys = temp.keys()
            sum_result = result
        else:
            sum_result += result

    return ODict(zip(keys, sum_result.tolist()))


def wav2spec(data: Dict) -> Dict:
    """"
    convert wav files into magnitude spectrogram
    :return:
    """
    spec = dict()
    for key, value in data.items():
        if len(key) < 4:
            value = value.squeeze()
            s = librosa.core.stft(value, **hp.kwargs_stft)
            spec[key] = np.abs(s)

    err = (np.abs(spec['out'] - spec['y']) ** 2) / (np.abs(spec['y']) ** 2 + 1e-6)
    err = np.maximum(10 * np.log10(err), -20.0)
    spec['err'] = err

    return spec


def draw_spectrogram(data: gen.TensArr, to_db=True, show=False, dpi=150, **kwargs):
    """

    :param data:
    :param to_db:
    :param show:
    :param err:
    :param dpi:
    :param kwargs: vmin, vmax
    :return:
    """

    data = data.squeeze()
    data = gen.convert(data, astype=ndarray)
    if to_db:
        data = librosa.amplitude_to_db(data)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(data,
              cmap=plt.get_cmap('CMRmap'),
              extent=(0, data.shape[1], 0, hp.sample_rate // 2),
              origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(ax.images[0], format='%+2.0f dB')

    fig.tight_layout()
    if show:
        fig.show()

    return fig


def draw_audio(data: gen.TensArr, fs: int, show=False, xlim=None, ylim=(-1, 1)):
    data = data.squeeze()
    data = gen.convert(data, astype=ndarray)
    t_axis = np.arange(len(data)) / fs
    if xlim is None:
        xlim = (0, t_axis[-1])

    fig, ax = plt.subplots(figsize=(xlim[1] * 10, 2), dpi=150)
    ax.plot(t_axis, data)
    ax.set_xlabel('time')
    ax.xaxis.set_major_locator(tckr.MultipleLocator(0.5))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.tight_layout()
    if show:
        fig.show()

    return fig


def principle_(angle):
    angle += np.pi
    angle %= (2 * np.pi)
    angle -= np.pi
    return angle


def reconstruct_wave(*args: ndarray,
                     n_iter=0, momentum=0., n_sample=-1,
                     **kwargs_istft) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag, phase) or (complex spectrogram,) or (mag,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param momentum: fast griffin-lim algorithm momentum
    :param n_sample: number of time samples of output wave
    :param kwargs_istft: kwargs for librosa.istft
    :return:
    """

    if len(args) == 1:
        if np.iscomplexobj(args[0]):
            spec = args[0].squeeze()
            mag = None
            phase = None
        else:
            spec = None
            mag = args[0].squeeze()
            # random initial phase
            phase = np.exp(2j * np.pi * np.random.rand(*mag.shape).astype(mag.dtype))
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError
    if not kwargs_istft:
        kwargs_istft = hp.kwargs_istft
        kwargs_stft = hp.kwargs_stft
    else:
        kwargs_stft = dict(n_fft=hp.n_fft, **kwargs_istft)

    spec_prev = 0
    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.istft(mag * np.exp(1j * phase), **kwargs_istft)
        spec_new = librosa.stft(wave, **kwargs_stft)

        phase = np.angle(spec_new - (momentum / (1 + momentum)) * spec_prev)
        spec_prev = spec_new

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.istft(spec, **kwargs_istft, **kwarg_len)

    return wave