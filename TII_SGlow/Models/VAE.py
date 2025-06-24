from utils import *

class Decoder(nn.Module):
    def __init__(self, hidden_dim, feat_out, pred_len, K):
        super(Decoder,self).__init__()

        self.hidden_dim = hidden_dim
        self.feat_out = feat_out
        self.pred_len = pred_len
        self.K = K
        self.pro1 = nn.Sequential(nn.Linear(self.hidden_dim * 2,self.hidden_dim), nn.ReLU())
        self.pro2 = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim), nn.ReLU())
        self.grucell = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        self.pro3 = nn.Linear(self.hidden_dim, feat_out)

    def forward(self,dec_in, Z):

        dec_h = torch.cat([dec_in, Z], dim=-1)

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
            forward_outputs = forward_outputs.view(self.pred_len,-1,self.K,self.hidden_dim)
        forward_outputs = self.pro3(forward_outputs)

        return forward_outputs

class VAE_tra(nn.Module):
    def __init__(self, args):
        super(VAE_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length

        self.encoder = nn.LSTM(input_size=self.feats_hidden,
                              hidden_size=self.feats_hidden, num_layers=2)

        self.fc_mu = nn.Sequential(
            nn.Linear(self.feats_hidden, self.feats_hidden),
            nn.ReLU())

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.feats_hidden, self.feats_hidden),
            nn.ReLU())

        self.fc_in = nn.Linear(self.feats_in, self.feats_hidden)
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.relu = nn.ReLU()
        self.decoder = Decoder(self.feats_hidden, self.feats_out,
                               self.pred_length, K=20)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs, iftrain=None):

        x = inputs[0][:self.obs_length,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)

        x = self.relu(self.fc_in(x))
        _, (h_n,_) = self.encoder(x)
        h = h_n[-1].unsqueeze(1).repeat(1,20,1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z, z)
        traj = reconstructed[-self.pred_length:]
        traj_loss = Traj_loss(traj, tar_y)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()

        return traj, traj_loss #+ 0.01 * kl_loss
