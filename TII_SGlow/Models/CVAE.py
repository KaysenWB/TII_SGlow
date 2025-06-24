from utils import *



class Decoder(nn.Module):
    def __init__(self, feats_hidden, feat_out, pred_len,fcond, K):
        super(Decoder,self).__init__()

        self.feats_hidden = feats_hidden
        self.feat_out = feat_out
        self.pred_len = pred_len
        self.K = K
        self.feats_cond = fcond
        self.pro1 = nn.Sequential(nn.Linear(self.feats_hidden*2 + self.feats_cond, self.feats_hidden),
                                  nn.ReLU())
        self.pro2 = nn.Sequential(nn.Linear(self.feats_hidden,self.feats_hidden),
                                  nn.ReLU())
        self.grucell = nn.GRUCell(input_size=self.feats_hidden, hidden_size=self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, feat_out)

    def forward(self, Z, dec, cond):

        dec_h = torch.cat([Z, dec, cond], dim=-1)

        forward_outputs = []
        forward_h = self.pro1(dec_h)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.pro2(forward_h)

        for t in range(self.pred_len):  # the last step is the goal, no need to predict
            forward_h= self.grucell(forward_input, forward_h)
            forward_input = self.pro2(forward_h)
            forward_outputs.append(forward_h)

        forward_outputs = torch.stack(forward_outputs, dim=0)
        if len(Z.shape) == 3:
            forward_outputs = forward_outputs.view(self.pred_len,-1,self.K,self.feats_hidden)
        forward_outputs = self.pro3(forward_outputs)

        return forward_outputs

class CVAE_tra(nn.Module):
    def __init__(self, args):
        super(CVAE_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.feats_cond = self.feats_hidden//2

        self.encoder = nn.LSTM(input_size=self.feats_hidden + self.feats_cond,
                              hidden_size=self.feats_hidden,num_layers=2)
        self.encoder2 = nn.LSTM(input_size=self.feats_cond,
                               hidden_size=self.feats_cond, num_layers=2)

        self.fc_mu = nn.Sequential(
            nn.Linear(self.feats_hidden, self.feats_hidden),
            nn.ReLU())

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.feats_hidden, self.feats_hidden),
            nn.ReLU())

        self.fc_in = nn.Linear(self.feats_in, self.feats_hidden)
        self.fc_in2 = nn.Linear(1, self.feats_cond)

        self.relu = nn.ReLU()

        self.decoder = Decoder(self.feats_hidden, self.feats_out,
                               self.pred_length, self.feats_cond, K=20)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs, iftrain=None):

        x = inputs[0][:self.obs_length,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)

        cond = inputs[0][:self.obs_length, :, -1:]
        cond = self.relu(self.fc_in2(cond))

        x = self.relu(self.fc_in(x))
        x = torch.cat([x, cond],dim=-1)
        _, (h_n,_) = self.encoder(x)
        h = h_n[-1].unsqueeze(1).repeat(1,20,1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        _, (c, _) = self.encoder2(cond)

        c = c[-1].unsqueeze(1).repeat(1,20,1)
        reconstructed = self.decoder(z, z, c)

        traj = reconstructed[-self.pred_length:]
        traj_loss = Traj_loss(traj, tar_y)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()

        return traj, traj_loss #+ 0.01 * kl_loss
