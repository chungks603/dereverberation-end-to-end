import torch.nn as nn
from hparams import hp
from waveunet.waveunet_utils import \
    DownBlock, UpBlock, BottleneckFC


class WaveUNet(nn.Module):
    def __init__(self, ch_double=False):
        super(WaveUNet, self).__init__()
        self.n_layers = hp.model['n_layers']
        self.ch_feature = hp.model['n_features']
        self.ch_interval = hp.model['ch_interval']

        # number of channels
        if not ch_double:
            n_channels = [self.ch_interval * i for i in range(1, self.n_layers + 1)]
        else:
            n_channels = [self.ch_interval * 2 ** i for i in range(0, self.n_layers)]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.middle = nn.ModuleList()

        for i in range(self.n_layers - 1):
            n_input = self.ch_feature if i == 0 else n_channels[i]

            self.encoder.append(
                DownBlock(n_input, n_channels[i], n_channels[i + 1])
            )

        for _ in range(2):
            self.middle = self.middle.append(
                BottleneckFC(n_channels[-1]),
            )

        for i in range(self.n_layers - 1):
            self.decoder.append(
                UpBlock(n_channels[-1 - i], n_channels[-2 - i])
            )

        self.out = nn.Conv1d(n_channels[0], 1, kernel_size=1, stride=1)

    def forward(self, x):
        skip = []
        out = x  # B, C, T

        # Down Blocks
        for i in range(self.n_layers - 1):
            out, skip_ = self.encoder[i](out)
            skip.append(skip_)

        # Bottleneck
        for i in range(2):
            out = self.middle[i](out)

        # Up Blocks
        for i in range(self.n_layers - 1):
            out = self.decoder[i](out, skip[-1 - i])

        # Out
        out = self.out(out)

        return out
