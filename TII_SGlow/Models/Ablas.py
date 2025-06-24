import torch

from .SGlow import *


class aff_conv_D_tra (nn.Module):
    def __init__(self, args):
        super(aff_conv_D_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = SGlow(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in * self.obs_length, self.feats_hidden),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in * self.pred_length, self.feats_hidden),
            nn.ReLU())

        self.decoder = Decoder(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]
        B = fut_enc.shape[1]

        # emb
        his_enc = self.fc_in_his(his_enc.permute(1, 0, 2).contiguous().view(B, -1))
        fut_enc = self.fc_in_fur(fut_enc.permute(1, 0, 2).contiguous().view(B, -1))
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        pred_traj = self.decoder(dec.permute(0,2,1))

        # cal loss
        loss_traj = self.loss.traj(pred_traj, fut)
        loss_dict = loss_flow + loss_traj * 0.5

        return pred_traj.permute(1,0,2,3), loss_dict


class D_tra (nn.Module):
    def __init__(self, args):
        super(D_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in * self.obs_length, self.feats_hidden),
            nn.ReLU())

        self.decoder = Decoder2(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        B = his_enc.shape[1]

        # emb
        enc = self.fc_in_his(his_enc.permute(1, 0, 2).contiguous().view(B, -1))

        pred_traj = self.decoder(enc)

        # cal loss
        loss_dict = self.loss.traj(pred_traj, fut)

        return pred_traj.permute(1,0,2), loss_dict


class aff_conv_tra (nn.Module):
    def __init__(self, args):
        super(aff_conv_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = SGlow(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in*self.obs_length, self.feats_hidden),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in*self.pred_length, self.feats_hidden),
            nn.ReLU())

        self.pred_next = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.loss = Loss(args)


    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]
        B = his_enc.shape[1]

        # emb
        his_enc = self.fc_in_his(his_enc.permute(1, 0, 2).contiguous().view(B, -1))
        fut_enc = self.fc_in_fur(fut_enc.permute(1, 0, 2).contiguous().view(B, -1))
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        traj = [dec.permute(0, 2, 1)]
        for i in range(self.pred_length):
            pred = self.pred_next(traj[-1])
            traj.append(pred)
        pred_traj = torch.stack(traj)[-self.pred_length:]
        pred_traj = self.fc_out(pred_traj)

        # cal loss
        loss_traj = self.loss.traj(pred_traj.permute(1,0,2,3), fut)
        loss_dict = loss_flow + loss_traj * 0.5

        return pred_traj, loss_dict


class E_aff_conv_tra (nn.Module):
    def __init__(self, args):
        super(E_aff_conv_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = SGlow(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())

        self.enco_his = nn.GRU(self.feats_hidden//2, self.feats_hidden)
        self.enco_fur = nn.GRU(self.feats_hidden//2, self.feats_hidden)

        self.pred_next = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.loss = Loss(args)


    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]

        # emb
        his_enc = self.enco_his(self.fc_in_his(his_enc))[0][-1]
        fut_enc = self.enco_fur(self.fc_in_fur(fut_enc))[0][-1]
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        traj = [dec.permute(0, 2, 1)]
        for i in range(self.pred_length):
            pred = self.pred_next(traj[-1])
            traj.append(pred)
        pred_traj = torch.stack(traj)[-self.pred_length:]
        pred_traj = self.fc_out(pred_traj)

        # cal loss
        loss_traj = self.loss.traj(pred_traj.permute(1,0,2,3), fut)
        loss_dict = loss_flow + loss_traj  * 0.5

        return pred_traj, loss_dict

class E_tra (nn.Module):
    def __init__(self, args):
        super(E_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())

        self.enco_his = nn.GRU(self.feats_hidden//2, self.feats_hidden)
        self.enco_fur = nn.GRU(self.feats_hidden//2, self.feats_hidden)

        self.pred_next = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.loss = Loss(args)


    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]

        # emb
        his_enc = self.enco_his(self.fc_in_his(his_enc))[0][-1]

        # predicting
        traj = [his_enc]
        for i in range(self.pred_length):
            pred = self.pred_next(traj[-1])
            traj.append(pred)
        pred_traj = torch.stack(traj)[-self.pred_length:]
        pred_traj = self.fc_out(pred_traj)

        # cal loss
        loss_traj = self.loss.traj(pred_traj.permute(1,0,2), fut)

        return pred_traj, loss_traj


class E_D_tra (nn.Module):
    def __init__(self, args):
        super(E_D_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())
        self.enco_his = nn.GRU(self.feats_hidden//2, self.feats_hidden)
        self.decoder = Decoder(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]

        # emb
        his_enc = self.enco_his(self.fc_in_his(his_enc))[0][-1]
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)

        # predicting
        pred_traj = self.decoder(his_enc.permute(0,2,1))

        # cal loss
        loss_traj = self.loss.traj(pred_traj, fut)

        return pred_traj.permute(1,0,2,3), loss_traj


class SGlow_aff(nn.Module):
    def __init__(self, args):
        super(SGlow_aff, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.affine.append(Affinecoupling_small(self.feats_hidden, feats_block))

    def forward(self, future_enc, history_enc):

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_a = self.affine[k](future_enc, history_enc)
            log_det = log_det + log_det_a

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)

        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample

class E_aff_D_tra (nn.Module):
    def __init__(self, args):
        super(E_aff_D_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = SGlow_aff(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())

        self.enco_his = nn.GRU(self.feats_hidden//2, self.feats_hidden)
        self.enco_fur = nn.GRU(self.feats_hidden//2, self.feats_hidden)

        self.decoder = Decoder(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]

        # emb
        his_enc = self.enco_his(self.fc_in_his(his_enc))[0][-1]
        fut_enc = self.enco_fur(self.fc_in_fur(fut_enc))[0][-1]
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        pred_traj = self.decoder(dec.permute(0,2,1))

        # cal loss
        loss_traj = self.loss.traj(pred_traj, fut)
        loss_dict = loss_flow + loss_traj * 0.5

        return pred_traj.permute(1,0,2,3), loss_dict


class SGlow_conv(nn.Module):
    def __init__(self, args):
        super(SGlow_conv, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling_none(self.feats_hidden, feats_block))

    def forward(self, future_enc):

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_c = self.convinv[k](future_enc)
            log_det = log_det + log_det_c

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)

        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].forward(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample


class E_conv_D_tra (nn.Module):
    def __init__(self, args):
        super(E_conv_D_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = SGlow_conv(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden//2),
            nn.ReLU())

        self.enco_his = nn.GRU(self.feats_hidden//2, self.feats_hidden)
        self.enco_fur = nn.GRU(self.feats_hidden//2, self.feats_hidden)

        self.decoder = Decoder(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]

        # emb
        his_enc = self.enco_his(self.fc_in_his(his_enc))[0][-1]
        fut_enc = self.enco_fur(self.fc_in_fur(fut_enc))[0][-1]
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        pred_traj = self.decoder(dec.permute(0,2,1))

        # cal loss
        loss_traj = self.loss.traj(pred_traj, fut)
        loss_dict = loss_flow + loss_traj * 0.5

        return pred_traj.permute(1,0,2,3), loss_dict


