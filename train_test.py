import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, List, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from apex import amp
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

import data_manager
import generic as gen
from adamwr import AdamW
from audio_utils import (calc_using_eval_module, calc_snrseg_time,
                         draw_spectrogram, wav2spec)
from emph import de_emphasis
from hparams import hp
from tbwriter import tb_writer
from utils import print_to_file, print_eval
from waveunet import waveunet_residual


class AverageMeter:
    """ Computes and stores the sum, count and the last value
    """

    def __init__(self,
                 init_factory: Callable = None,
                 init_value: Any = 0.,
                 init_count=0):
        self.init_factory: Callable = init_factory
        self.init_value = init_value

        self.reset(init_count)

    def reset(self, init_count=0):
        if self.init_factory:
            self.last = self.init_factory()
            self.sum = self.init_factory()
        else:
            self.last = self.init_value
            self.sum = self.init_value
        self.count = init_count

    def update(self, value, n=1):
        self.last = value
        self.sum += value
        self.count += n

    def get_average(self):
        try:
            self.avg = self.sum / self.count
            return self.avg
        except ZeroDivisionError as z:
            print(z)


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hp):
        # TODO: model, criterion
        self.loss = nn.L1Loss(reduction='none')
        self.model = waveunet_residual.WaveUNet(ch_double=hp.model['ch_double'])

        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(),
                               lr=hp.learning_rate,
                               weight_decay=hp.weight_decay, )

        # device
        device_for_summary = self._init_device(hp.device, hp.out_device)

        # summary
        self.writer = SummaryWriter(log_dir=hp.logdir)
        path_summary = Path(self.writer.log_dir, 'summary.txt')
        if not path_summary.exists():
            print_to_file(path_summary,
                          summary,
                          (self.model, hp.dummy_input),
                          dict(device=device_for_summary)
                          )

        # save hyper-parameters
        path_hp = Path(self.writer.log_dir, 'hp.txt')
        if not path_hp.exists():
            print_to_file(path_hp, hp.print_params)

        # evaluation metric
        self.metric = ['SegSNR', 'fwSegSNR', 'PESQ', 'STOI']

    def _init_device(self, device, out_device) -> str:
        if device == 'cpu':
            self.device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return 'cpu'

        # device type
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) == int:
                device = device
            else:
                device = [int(d[-1]) for d in device]

        # out_device type
        if type(out_device) == int:
            out_device = torch.device(f'cuda:{out_device}')
        else:
            out_device = torch.device(out_device)
        self.device = torch.device(f'cuda:{device[0]}')
        self.out_device = out_device

        if len(device) > 1:
            self.model, self.optimizer = amp.initialize(self.model.cuda(self.device), self.optimizer,
                                                        opt_level="O1",
                                                        cast_model_type=None,
                                                        patch_torch_functions=True,
                                                        keep_batchnorm_fp32=None,
                                                        master_weights=None,
                                                        loss_scale="dynamic")

            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])
        else:
            self.str_device = str(self.device)

        self.model.cuda(device[0])
        torch.cuda.set_device(device[0])

        return 'cuda'

    def preprocess(self, data: Dict[str, Tensor]) -> Tuple:
        # Take out the data for training model
        xs = data['x']
        ys = data['y']
        T_ys = data['T_ys']

        xs = xs.to(self.device)
        ys = ys.to(self.out_device)
        T_ys = T_ys.to(self.device)

        return xs, ys, T_ys

    @torch.no_grad()
    def postprocess(self, data: Dict,
                    dataset: data_manager.CustomDataset) -> List[Dict]:
        samples, outs = [], []

        # speech de-normalization, de-emphasis
        xs = data['x'].permute(0, 2, 1)  # B, T, C
        xs = dataset.norm_modules['x'].denormalize_(xs)
        xs = de_emphasis(gen.convert(xs[..., 0], astype=np.ndarray))

        ys = data['y']
        ys = dataset.norm_modules['y'].denormalize_(ys)
        ys = de_emphasis(gen.convert(ys, astype=np.ndarray))

        outs = data['out']
        outs = dataset.norm_modules['y'].denormalize_(outs)
        outs = de_emphasis(gen.convert(outs, astype=np.ndarray))

        data['x'], data['y'], data['out'] = xs, ys, outs

        T_xs = data['T_xs'].int()
        T_ys = data['T_ys'].int()

        # Dict[List] -> List[Dict]
        for idx in range(len(data['x'])):
            sample = dict()
            T_x, T_y = T_xs[idx], T_ys[idx]
            sample['T_xs'], sample['T_ys'] = T_x, T_y

            for key, value in data.items():
                value = value[idx]
                if key == 'x':
                    value = value[..., :T_x]
                    value = np.nan_to_num(value, posinf=1., neginf=-1.)
                    value = np.asfortranarray(np.clip(value, -1, 1))
                elif len(key) > 3:
                    pass
                else:
                    value = value[..., :T_y]
                    value = np.nan_to_num(value, posinf=1., neginf=-1.)
                    value = np.asfortranarray(np.clip(value, -1, 1))
                sample[key] = value
            samples.append(sample)

        return samples

    def calc_loss(self, out: Tensor, y: Tensor, T_ys: Tensor) -> Tensor:
        loss_batch = self.loss(out, y)
        loss = torch.zeros(1, device=loss_batch.device)
        for T, loss_sample in zip(T_ys, loss_batch):
            loss += torch.sum(loss_sample[:T]) / T

        loss = torch.log10(loss + 1)
        return loss.float()

    @staticmethod
    def evaluate(data: List[Dict], key: str):
        ys = [item['y'] for item in data]
        ests = [item[key] for item in data]
        T_ys = [item['T_ys'] for item in data]

        segsnr_val = calc_snrseg_time(ys, ests, l_frame=512, l_hop=256, T_ys=T_ys)
        eval_val = calc_using_eval_module(ys, ests, T_ys)

        eval_result = np.array([segsnr_val, *eval_val.values()])

        return eval_result

    # Running model for train and validation.
    def run(self, loader, mode: str, epoch: int):
        if mode == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        avg_loss = AverageMeter(float)
        avg_eval = np.array([0., 0., 0.])
        avg_grad_norm = 0.

        pbar = tqdm(loader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)
        for i_batch, data in enumerate(pbar):
            # get data - xs, ys: B, C, T
            xs, ys, T_ys = self.preprocess(data)

            # Forward
            outs = self.model(xs).squeeze()
            outs = outs[..., :ys.shape[-1]]

            # Loss calculation
            loss = self.calc_loss(outs, ys, T_ys)

            # Backward
            if mode == 'train':
                self.optimizer.zero_grad()
                if torch.isfinite(loss):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), hp.thr_clip_grad)
                    avg_grad_norm += grad_norm
                    self.writer.add_scalar('loss/grad', avg_grad_norm / float(pbar.total), epoch)

                    self.optimizer.step()

            # Data upload to the tensorboard (audio, spectrogram, scalars)
            if mode == 'valid' and i_batch == 0:
                data['out'] = outs.cpu()
                data = self.postprocess(data, loader.dataset)
                sample = data[0]
                tb_writer(self.writer, sample, mode, epoch)

                if epoch == 0:
                    global eval_xs
                    eval_xs = self.evaluate(data, 'x') / len(data)  # eval for input

                eval_outs = self.evaluate(data, 'out') / len(data)  # eval for model output
                for metric, eval_out, eval_x in zip(self.metric, eval_outs, eval_xs):
                    self.writer.add_scalar(f'eval/{metric}_out', eval_out, epoch)
                    self.writer.add_scalar(f'eval/{metric}_x', eval_x, epoch)

            eval_outs = 0.

            if torch.isfinite(loss):
                avg_loss.update(loss.item(), len(T_ys))
                pbar.set_postfix_str(f'loss:{avg_loss.get_average():.2e}')

            avg_eval += eval_outs

        avg_eval /= len(loader.dataset)

        self.writer.close()

        return avg_loss.get_average(), avg_eval

    @torch.no_grad()
    def test(self, loader, epoch: int):
        self.model.eval()
        state_dict = torch.load(Path(self.writer.log_dir, f'{epoch}.pt'))
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        path_test_result = Path(self.writer.log_dir, f'test_{epoch}')
        os.makedirs(path_test_result, exist_ok=True)

        avg_eval_out, avg_eval_x = 0., 0.

        pbar = tqdm(loader, desc=f'test {epoch:3d}', postfix='-', dynamic_ncols=True)
        for i_batch, data in enumerate(pbar):
            xs, ys, T_ys = self.preprocess(data)

            outs = self.model(xs).squeeze()

            data['out'] = outs
            data = self.postprocess(data, loader.dataset)

            sample = data[0]
            spec = wav2spec(sample)

            fig_x = draw_spectrogram(spec['x'], dpi=300, **dict(vmin=-50, vmax=20))
            fig_y = draw_spectrogram(spec['y'], dpi=300, **dict(vmin=-50, vmax=20))
            fig_out = draw_spectrogram(spec['out'], dpi=300, **dict(vmin=-50, vmax=20))
            fig_err = draw_spectrogram(spec['err'], dpi=300, to_db=False, **dict(vmin=-20, vmax=20))

            fig_x.savefig(path_test_result / str(sample["fname"][:-4] + '_x.png'))
            fig_y.savefig(path_test_result / str(sample['fname'][:-4] + '_y.png'))
            fig_out.savefig(path_test_result / str(sample['fname'][:-4] + '_out.png'))
            fig_err.savefig(path_test_result / str(sample['fname'][:-4] + '_err.png'))

            plt.close('all')

            eval_outs = self.evaluate(data, 'out')
            avg_eval_out += eval_outs

            eval_xs = self.evaluate(data, 'x')
            avg_eval_x += eval_xs

        avg_eval_out = avg_eval_out / len(loader.dataset)
        avg_eval_x = avg_eval_x / len(loader.dataset)

        return avg_eval_out, avg_eval_x


def main(test_epoch: int):
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hp)
    if test_epoch == -1:
        runner = Runner(hp)

        # TODO: add all the evaluation metrics
        dict_loss = dict(loss=['Multiline', ['loss/train', 'loss/valid']])
        dict_eval = dict(PESQ=['Multiline', ['eval/PESQ_out', 'eval/PESQ_x']],
                         STOI=['Multiline', ['eval/STOI_out', 'eval/STOI_x']],
                         SegSNR=['Multiline', ['eval/SegSNR_out', 'eval/SegSNR_x']],
                         fwSegSNR=['Multiline', ['eval/fwSegSNR_out', 'eval/fwSegSNR_x']])
        runner.writer.add_custom_scalars(dict(train=dict_loss, valid=dict_eval))

        epoch = 0
        test_epoch_or_zero = 0
        print(f'Training on {runner.str_device}')
        for epoch in range(hp.num_epochs):
            # training
            train_loss, train_eval = runner.run(train_loader, 'train', epoch)
            if train_loss is not None:
                if torch.isfinite(torch.tensor(train_loss)):
                    runner.writer.add_scalar('loss/train', train_loss, epoch)

            # checkpoint save
            torch.save(runner.model.module.state_dict(), Path(runner.writer.log_dir, f'{epoch}.pt'))

            # validation
            valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)
            if valid_loss is not None:
                if torch.isfinite(torch.tensor(valid_loss)):
                    runner.writer.add_scalar('loss/valid', valid_loss, epoch)

        print('Training Finished')
        test_epoch = test_epoch_or_zero if test_epoch_or_zero > 0 else epoch
    else:
        runner = Runner(hp)

    # test
    test_eval_outs, test_eval_xs = runner.test(test_loader, test_epoch)

    # TODO: write test result
    str_metric = ['SegSNR', 'fwSegSNR', 'PESQ', 'STOI']
    print_eval_outs, print_eval_xs = dict(), dict()
    for k, eval_out, eval_x in zip(str_metric, test_eval_outs, test_eval_xs):
        print_eval_outs[k] = eval_out
        print_eval_xs[k] = eval_x

    print(f'Test - Input Eval: {print_eval_xs}')
    print(f'Test - Out Eval: {print_eval_outs}')

    path_eval = Path(hp.logdir, f'test_{test_epoch}', 'test_eval.txt')
    if not path_eval.exists():
        print_to_file(path_eval,
                      print_eval,
                      (print_eval_xs, print_eval_outs)
                      )

    runner.writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', type=int, default=-1)

    args = hp.parse_argument(parser)
    test_epoch = args.test
    if test_epoch == -1:
        # check overwrite or not
        if list(Path(hp.logdir).glob('events.out.tfevents.*')):
            while True:
                s = input(f'"{hp.logdir}" already has tfevents. continue? (y/n)\n')
                if s.lower() == 'y':
                    shutil.rmtree(hp.logdir)
                    os.makedirs(hp.logdir)
                    break
                elif s.lower() == 'n':
                    exit()

    main(test_epoch)
