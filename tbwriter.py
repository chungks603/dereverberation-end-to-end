"""
Tensorboard summary writer for the first batch of the validation set
"""
import numpy as np
from typing import Dict
from torch.utils.tensorboard.writer import SummaryWriter
from hparams import hp
from audio_utils import draw_audio, draw_spectrogram, wav2spec


def tb_writer(writer: SummaryWriter, sample: Dict, mode: str, epoch: int):
    """

    :param writer:
    :param mode:
    :param epoch:
    :param sample: one sample of the first batch in the validation set
    :return:
    """
    spec = wav2spec(sample)

    keys = ['x', 'y', 'out', 'err']
    for k in keys:
        if not k == 'err':
            writer.add_audio(f'{mode}/wav_{k}', sample[f'{k}'], epoch, hp.sample_rate)

            waveform = draw_audio(sample[f'{k}'], hp.sample_rate)
            writer.add_figure(f'{mode}/waveform_{k}', waveform, epoch)

            fig = draw_spectrogram(spec[f'{k}'], **hp.kwargs_spec)

        else:
            fig = draw_spectrogram(spec[f'{k}'], to_db=False, **dict(vmin=-20, vmax=20))

        writer.add_figure(f'{mode}/spec_{k}', fig, epoch)
