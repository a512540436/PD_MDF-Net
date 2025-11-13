import random
import numpy as np
import pywt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy 的随机种子
random.seed(seed)


# ========== 特征变换函数 ==========
def to_freq_domain(x):
    """FFT频域特征"""
    ffted = torch.fft.rfft(x, dim=-1)
    return torch.abs(ffted)


def to_wavelet_domain(x, wave='db4', level=2):
    coeffs = []
    for i in range(x.shape[1]):
        cA, cD = pywt.dwt(x[:, i, :].cpu().numpy(), wave)
        arr = np.concatenate([cA, cD], axis=-1)
        # 自动对齐到原长度
        arr = arr[:, :x.shape[-1]] if arr.shape[-1] > x.shape[-1] else \
            np.pad(arr, ((0, 0), (0, x.shape[-1] - arr.shape[-1])))
        coeffs.append(torch.tensor(arr))
    return torch.stack(coeffs, dim=1).to(x.device)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.GELU1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.GELU2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.GELU1, self.dropout1,
                                 self.conv2, self.chomp2, self.GELU2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.GELU = nn.GELU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.GELU(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y1 = self.network(x)
        y2 = y1.permute(0, 2, 1)
        out = self.linear(y2).squeeze(-1)
        return out


class TTCN(nn.Module):
    def __init__(self, in_channels, num_channels=(64, 128, 256), kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TCN(
            num_inputs=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.attn = nn.Linear(num_channels[-1], 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        y = self.tcn.network(x)
        y = y.permute(0, 2, 1)

        scores = self.attn(y)
        weights = F.softmax(scores, dim=1)
        y = (y * weights).sum(dim=1)

        y = self.gelu(y)
        return y  # [B, hidden]


class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing = nn.Linear(num_patches, num_patches)

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.channel_norm(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


#
# class FreqWaveMixer(nn.Module):
#     def __init__(self, n_channels, in_features, hidden_dim=64, out_dim=32, num_blocks=2):
#         super().__init__()
#         self.proj = nn.Linear(in_features, hidden_dim)
#         self.blocks = nn.Sequential(
#             *[MixerBlock(n_channels, hidden_dim) for _ in range(num_blocks)]
#         )
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, out_dim)
#
#     def forward(self, x):
#         x = self.proj(x)
#         x = self.blocks(x)
#         x = self.norm(x.mean(dim=1))
#         return self.fc(x)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        out = (x * weights).sum(dim=1)
        return out


# ===== Mixer Block =====
class CmixBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.token_norm = nn.BatchNorm1d(num_tokens)
        self.token_mixing = nn.Linear(num_tokens, num_tokens)

        self.channel_norm = nn.BatchNorm1d(hidden_dim)
        self.channel_mixing = nn.Sequential(
            nn.Conv1d(hidden_dim, int(hidden_dim * 1.5), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(int(hidden_dim * 1.5), hidden_dim, kernel_size=1)
        )

    def forward(self, x):
        B, T, H = x.shape

        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        y = x.transpose(1, 2)
        y = self.channel_norm(y)
        y = self.channel_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        return x


# ===== Branch =====
class Cmix(nn.Module):
    def __init__(self, output_dim=64, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.proj = None  #
        self.blocks = None
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, F = x.shape
        if self.proj is None:
            self.proj = nn.Conv1d(F, self.hidden_dim, kernel_size=1).to(x.device)
            self.blocks = nn.Sequential(
                *[CmixBlock(T, self.hidden_dim) for _ in range(self.num_blocks)]
            ).to(x.device)

        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x)


class MDFNet(nn.Module):
    def __init__(self, n_channels=32, shape_T=20,
                 n_classes=2, conv_channels=(64, 128, 64), fc_hidden=64):
        super().__init__()
        self.out_dim = n_classes
        self.n_channels = n_channels
        self.shape_T = shape_T

        def get_transformed_shapes_simple(T):
            T_fft = T // 2 + 1
            dummy = [0] * T
            cA, cD = pywt.dwt(dummy, 'db4')
            return T_fft, len(cA) + len(cD)

        self.freq_t, self.wave_t = get_transformed_shapes_simple(self.shape_T)

        self.branch_time = TTCN(n_channels, conv_channels)
        self.branch_freq = Cmix(output_dim=conv_channels[-1])
        self.branch_wave = Cmix(output_dim=conv_channels[-1])

        feature_dim = conv_channels[-1] * 3
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x):
        x_time = x[:, :, :self.shape_T]
        x_freq = x[:, :, self.shape_T:self.shape_T + self.freq_t]
        x_wave = x[:, :, self.shape_T + self.freq_t:]

        feat_time = self.branch_time(x_time)
        feat_freq = self.branch_freq(x_freq)
        feat_wave = self.branch_wave(x_wave)

        feats = torch.cat([feat_time, feat_freq, feat_wave], dim=1)
        return self.classifier(feats)


class MultiDomainEEGNet_TF(nn.Module):
    def __init__(self, n_channels=32, shape_T=60,
                 n_classes=2, conv_channels=(64, 128), fc_hidden=64):
        super().__init__()
        self.out_dim = n_classes
        self.n_channels = n_channels
        self.shape_T = shape_T

        def get_transformed_shapes_simple(T):
            T_fft = T // 2 + 1
            dummy = [0] * T
            cA, cD = pywt.dwt(dummy, 'db4')
            return T_fft, len(cA) + len(cD)

        self.freq_t, self.wave_t = get_transformed_shapes_simple(self.shape_T)
        # print(self.freq_t, self.wave_t)
        # 分支
        self.branch_time = TTCN(n_channels, conv_channels)
        self.branch_freq = Cmix(output_dim=conv_channels[-1])

        feature_dim = conv_channels[-1] * 3
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x):
        x_time = x[:, :, :self.shape_T]
        x_freq = x[:, :, self.shape_T:self.shape_T + self.freq_t]

        feat_time = self.branch_time(x_time)
        feat_freq = self.branch_freq(x_freq)

        feats = torch.cat([feat_time, feat_freq], dim=1)
        return self.classifier(feats)


class MultiDomainEEGNet_vTW(nn.Module):
    def __init__(self, n_channels=32, shape_T=60,
                 n_classes=2, conv_channels=(64, 128), fc_hidden=64):
        super().__init__()
        self.out_dim = n_classes
        self.n_channels = n_channels
        self.shape_T = shape_T

        def get_transformed_shapes_simple(T):
            T_fft = T // 2 + 1
            dummy = [0] * T
            cA, cD = pywt.dwt(dummy, 'db4')
            return T_fft, len(cA) + len(cD)

        self.freq_t, self.wave_t = get_transformed_shapes_simple(self.shape_T)

        # 分支
        self.branch_time = TTCN(n_channels, conv_channels)
        self.branch_wave = Cmix(output_dim=conv_channels[-1])

        feature_dim = conv_channels[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x):
        x_time = x[:, :, :self.shape_T]
        x_wave = x[:, :, self.shape_T + self.freq_t:]

        feat_time = self.branch_time(x_time)
        feat_wave = self.branch_wave(x_wave)

        feats = torch.cat([feat_time, feat_wave], dim=1)
        return self.classifier(feats)
