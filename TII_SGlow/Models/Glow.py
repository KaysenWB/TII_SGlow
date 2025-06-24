from .Flows import *


class Glow(nn.Module):
    def __init__(self, args):
        super(Glow, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.instancenorm = nn.ModuleList()
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))
            self.instancenorm.append(GroupInstanceNorm(feats_block, group_len=4))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.instancenorm[k](future_enc)
            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_a + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)
            z_sample = self.instancenorm[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample



class Glow_tra (nn.Module):
    def __init__(self, args):
        super(Glow_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = Glow(args)

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