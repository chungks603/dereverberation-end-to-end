import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hp


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class WaveUNet(nn.Module):
    def __init__(self):
        super(WaveUNet, self).__init__()
        self.n_layers = hp.model['n_layers']
        self.ch_feature = hp.model['n_features']
        self.ch_interval = hp.model['ch_interval']

        # number of encoder channels
        encoder_in = [self.ch_feature] + [(i + 1) * self.ch_interval for i in range(self.n_layers - 1)]
        encoder_out = [(i + 1) * self.ch_interval for i in range(self.n_layers)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in[i],
                    channel_out=encoder_out[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.ch_interval,
                      (self.n_layers + 1) * self.ch_interval, 15, stride=1, padding=7),
            nn.BatchNorm1d((self.n_layers + 1) * self.ch_interval),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # number of decoder channels
        decoder_in = [(2 * i + 1) * self.ch_interval for i in range(1, self.n_layers + 1)]
        decoder_in = decoder_in[::-1]
        decoder_out = [i * self.ch_interval for i in range(self.n_layers, 0, -1)]

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in[i],
                    channel_out=decoder_out[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(self.ch_feature + self.ch_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        skip = []
        out = x

        # Down Sampling
        for i in range(self.n_layers):
            out = self.encoder[i](out)
            skip.append(out)
            # batch, channel, T // 2
            out = out[:, :, ::2]

        out = self.middle(out)

        # Up Sampling
        for i in range(self.n_layers):
            # batch, channel, T * 2
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=True)

            # Skip Connection
            out = torch.cat([out, skip[self.n_layers - 1 - i]], dim=1)
            out = self.decoder[i](out)

        out = torch.cat([out, x], dim=1)
        out = self.out(out)
        return out
