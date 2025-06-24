from .Flows import *


class Affinecoupling_small(nn.Module):

    def __init__(self, feats_hidden, feats_block):
        super(Affinecoupling_small, self).__init__()
        self.z_embed = nn.Sequential(nn.Linear(feats_hidden + feats_block//2, 2 * feats_hidden),
                                     nn.GELU())

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


class SGlow(nn.Module):
    def __init__(self, args):
        super(SGlow, self).__init__()

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
            self.affine.append(Affinecoupling_small(self.feats_hidden, feats_block))

    def forward(self, future_enc, history_enc):

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ =  log_det_a + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)

        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample



class SGlow_tra (nn.Module):
    def __init__(self, args):
        super(SGlow_tra, self).__init__()

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

        self.decoder = Decoder(args)
        self.loss = Loss(args)

        '''gl = sum(p.numel() for p in self.glow.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in self.enco_his.parameters() if p.requires_grad)
        fc_in = sum(p.numel() for p in self.fc_in_his.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        Para = gl + enc *2 + fc_in *2 + dec'''

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

