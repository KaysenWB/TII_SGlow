
import torch
import torch.nn as nn
import numpy as np
from utils import Traj_loss


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result




class ConvTemporalGraphical(nn.Module):

    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size),
            padding=(t_padding),
            stride=(t_stride),
            dilation=(t_dilation),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        x = x.contiguous()
        return x, A


class st_gcn(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()


    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class ST_GCN(nn.Module):
    def __init__(self, args):
        super(ST_GCN, self).__init__()
        self.args = args
        self.n_stgcnn = 1
        self.n_txpcnn = 5
        self.kernel_size = 3
        self.feats_in = args.feats_in
        self.feats_hidden = args.feats_hidden
        self.feats_out = args.feats_out
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(self.feats_hidden, 5,
                        (self.kernel_size, self.obs_length)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(5, 5,
                            (self.kernel_size, self.obs_length)))
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(self.obs_length, self.pred_length, self.kernel_size, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(self.pred_length, self.pred_length, self.kernel_size, padding=1))
        self.tpcnn_ouput = nn.Conv2d(self.pred_length, self.pred_length, self.kernel_size, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.fc_in = nn.Sequential(nn.Linear(self.feats_in, self.feats_hidden),nn.ReLU())

    def forward(self, inputs, iftrain):
        # v, a
        x = inputs[0][:self.obs_length, :, :self.feats_in]#.unsqueeze(0).permute(0, 3, 1, 2)
        tra_y = inputs[0][self.obs_length:, :, :self.feats_in]
        a = inputs[2][:self.obs_length, :, :]

        v = self.fc_in(x).unsqueeze(0).permute(0, 3, 1, 2)
        for k in range(self.n_stgcnn):  # 1(B), 5(F), 16(T), 106(N)
            v, a = self.st_gcns[k](v, a)

        v = v.permute(0, 2, 1, 3)
        v = self.prelus[0](self.tpcnns[0](v))  # 1(B), 16(T), 5(F), 106(N)
        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        v = self.tpcnn_ouput(v)
        v = v.squeeze().permute(0, 2, 1)

        loss = bivariate_loss(v, tra_y)
        #loss = Traj_loss(v, tra_y)
        return v[:,:,:2], loss

