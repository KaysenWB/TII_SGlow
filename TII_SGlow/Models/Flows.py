
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import nflows.utils.typechecks as check
from math import log, pi

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, n_of_group = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = n_of_group * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class GroupInstanceNorm(nn.Module):
    def __init__(self, features=None, group_len=5):
        """
        Args:
            group_len: the len of group
            tensor: shape of (B, L, K), K for the number of trajectory
        Returns: shape of (B, L, K)
        """
        if not check.is_positive_int(group_len):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()
        self.group_len = group_len
        self.features = features
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        # self.scale = nn.Parameter(torch.zeros(group_len))
        if features is not None:
            self.log_scale = nn.Parameter(torch.zeros(features, group_len))
            self.shift = nn.Parameter(torch.zeros(features, group_len))
        else:
            self.log_scale = nn.Parameter(torch.zeros(group_len))
            self.shift = nn.Parameter(torch.zeros(group_len))

    @property
    def scale(self):
        return torch.exp(torch.clamp(self.log_scale, max=1e10))

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.training and not self.initialized:
            self._initialize(inputs)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        log_det = torch.sum(self.log_scale)

        return outputs, log_det

    def reverse(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        return outputs

    def _initialize(self, inputs):
        """Data-dependent initialization"""
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / (std + 1e-6)).mean(dim=0)
            std = std.mean(-2)
            mu = mu.mean(-2)
            self.log_scale.data = -torch.log(torch.clamp(std, 1e-10, 1e10))
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.randn(c, c))[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z):
        # shape
        B, F, K, = z.shape
        W = self.conv.weight.squeeze()
        log_det_W = B * K * torch.logdet(W)
        z = self.conv(z)
        return z, log_det_W

    def reverse (self, z):

        W = self.conv.weight.squeeze()

        if not hasattr(self, 'W_inverse'):
            # Reverse computation
            bias = torch.diag(torch.zeros_like(W[0])+1e-6)
            W_inverse = (W+bias).float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor':
                W_inverse = W_inverse.half()
        z = F.conv1d(z, W_inverse, bias=None, stride=1, padding=0)
        return z




class Affinecoupling(nn.Module):

    def __init__(self, feats_hidden, feats_block):
        super(Affinecoupling, self).__init__()
        self.z_embed = nn.Sequential(nn.Linear(feats_hidden + feats_block//2, 2 * feats_hidden),
                                     nn.GELU(),
                                     nn.Linear(2 * feats_hidden, feats_hidden*2),
                                     nn.GELU(),)
        for m in self.z_embed.modules():
            if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.zero_()
        end = nn.Linear(feats_hidden*2, feats_block)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.tanh = nn.Tanh()
        self.end = end

    def forward(self, z, history_enc):

        z_0, z_1 = z.chunk(2, 1)
        h_x = torch.cat([history_enc, z_0], 1)
        h_x = self.z_embed(h_x.permute(0, 2, 1))
        h_x = self.end(h_x).permute(0, 2, 1)  # 5.22
        h_x = self.tanh(h_x)
        log_s, b = h_x.chunk(2, 1)
        z_1 = torch.exp(log_s) * z_1 + b
        z = torch.cat([z_0, z_1], 1)
        log_det = torch.sum(log_s)

        return z, log_det

    def reverse(self, z_sample, history_enc):

        z_sample_0, z_sample_1 = z_sample.chunk(2, 1)
        h_x = torch.cat([history_enc, z_sample_0], 1)
        h_x = self.z_embed(h_x.permute(0, 2, 1))
        h_x = self.end(h_x).permute(0, 2, 1)  # 5.22
        h_x = self.tanh(h_x)
        s, b = h_x.chunk(2, 1)
        z_sample_1 = (z_sample_1 - b) / torch.exp(s)
        z_sample = torch.cat([z_sample_0, z_sample_1], 1)

        return z_sample


class Affinecoupling_none(nn.Module):

    def __init__(self, feats_hidden, feats_block):
        super(Affinecoupling_none, self).__init__()
        self.z_embed = nn.Sequential(nn.Linear(feats_hidden + feats_block, feats_block),
                                     nn.GELU())

    def forward(self, z_sample, history_enc):

        h_x = torch.cat([history_enc, z_sample], 1)
        h_x = self.z_embed(h_x.permute(0, 2, 1))
        z_sample = h_x.permute(0, 2, 1)  # 5.22

        return z_sample


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.K = args.K

    def glow(self, z, log_det):

        log_p = -0.5 * log(2*pi) - 0.5 * (z ** 2)
        log_p_sum = torch.sum(log_p)
        loss = - (log_p_sum  + log_det)
        loss = loss/(z.size(0)*z.size(1)*z.size(2))

        return loss

    def traj(self, pred_traj, fur):
        if pred_traj.dim() ==4:
            fur = fur.unsqueeze(2).repeat(1, 1, self.K, 1)
            traj_rmse = torch.sqrt(torch.sum((pred_traj - fur) ** 2, dim=-1) + 1e-8).sum(dim=1)
            best_idx = torch.argmin(traj_rmse, dim=1)
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        else:
            traj_rmse = torch.sqrt(torch.sum((pred_traj - fur) ** 2, dim=-1) + 1e-8).sum(dim=1)
            loss_traj = traj_rmse.mean()

        return loss_traj

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.feats_hidden = args.feats_hidden
        self.feats_out = args.feats_out
        self.pred_len = args.pred_length

        self.pro1 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden * 2, self.feats_out)
        self.gru = nn.GRUCell(self.feats_hidden, self.feats_hidden)
        self.relu = nn.ReLU()


    def forward (self, dec_h):

        B, K, F = dec_h.shape
        forward_output = []
        forward_h = self.relu(self.pro1(dec_h)).view(-1, F)
        forward_input = self.relu(self.pro2(forward_h))

        for t in range(self.pred_len):
            forward_h = self.gru(forward_input, forward_h)
            forward_input = self.relu(self.pro2(forward_h))
            forward_traj = self.pro3(torch.cat([forward_input, dec_h.reshape(-1, F)], 1))
            forward_output.append(forward_traj.view(-1, K, forward_traj.shape[-1]))

        pred_traj = torch.stack(forward_output, dim=1)

        return pred_traj


class Decoder2(nn.Module):
    def __init__(self, args):
        super(Decoder2, self).__init__()
        self.feats_hidden = args.feats_hidden
        self.feats_out = args.feats_out
        self.pred_len = args.pred_length

        self.pro1 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden * 2, self.feats_out)
        self.gru = nn.GRUCell(self.feats_hidden, self.feats_hidden)
        self.relu = nn.ReLU()


    def forward (self, dec_h):

        forward_output = []
        forward_h = self.relu(self.pro1(dec_h))
        forward_input = self.relu(self.pro2(forward_h))

        for t in range(self.pred_len):
            forward_h = self.gru(forward_input, forward_h)
            forward_input = self.relu(self.pro2(forward_h))
            forward_traj = self.pro3(torch.cat([forward_input, dec_h], 1))
            forward_output.append(forward_traj)

        pred_traj = torch.stack(forward_output, dim=1)

        return pred_traj