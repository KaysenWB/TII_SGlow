from utils import *
import torch.optim.lr_scheduler as lr_sche



class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.feats_in = args.feats_in
        self.feats_out = args.feats_out
        self.feats_hidden = args.feats_hidden
        self.pred_length = args.pred_length
        self.obs_length = args.obs_length
        self.K = args.K
        self.device = args.device

        self.fc_in = nn.Linear(self.feats_in, self.feats_hidden)
        self.encoder = nn.LSTM(self.feats_hidden, self.feats_hidden, 4)

        self.dec =  nn.LSTM(self.feats_hidden, self.feats_hidden)
        self.fc_h = nn.Sequential(nn.Linear(self.feats_hidden * 2, self.feats_hidden),
                                  nn.ReLU())
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out,)
        self.norm = nn.BatchNorm1d(self.feats_hidden)
        self.relu = nn.ReLU()


    def forward(self, cond):

        L, B, _ = cond.shape
        cond = self.encoder(self.fc_in(cond))[0][-1]
        cond = cond.unsqueeze(1).repeat(1, self.K, 1)
        z_samples = torch.randn(cond.shape).to(self.device)

        enc = torch.cat([z_samples, cond], dim=-1)
        enc = self.fc_h(enc).unsqueeze(0).view(1, -1, self.feats_hidden).repeat(self.pred_length, 1, 1)
        out = self.dec(enc)[0]
        out = self.fc_out(out).view(L, B, self.K, self.feats_out)

        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.remains_dict = {'16':160, '32':416, '48':672, '64':928}
        self.remains = self.remains_dict[str(self.args.obs_length)]

        self.fc_in = nn.Sequential(
            nn.Linear(self.feats_in, self.feats_hidden),
            nn.ReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self.remains, self.remains//2),
            nn.ReLU(),
            nn.Linear(self.remains//2, self.feats_out),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.feats_hidden, self.feats_hidden // 2, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(self.feats_hidden // 2, self.feats_hidden // 8, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
        )


    def forward(self, x):

        L, B, K , _ = x.shape
        x = self.fc_in(x).permute(1, 3, 0, 2)
        out = self.conv(x)
        out = out.view(B, -1)
        out = self.fc_out(out)

        return out


class GAN_tra(nn.Module):
    def __init__(self, args):
        super(GAN_tra, self).__init__()
        self.args = args
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.feats_in = args.feats_in
        self.feats_out = args.feats_out
        self.feats_hidden = args.feats_hidden
        self.device = args.device
        self.K = args.K

        self.fc_in = nn.Linear(self.feats_in, self.feats_hidden)
        self.encoder = nn.LSTM(self.feats_hidden,self.feats_hidden)

        self.generator = Generator(args)
        self.discriminator = Discriminator(args)
        self.loss_fn = nn.BCELoss()
        self.loss_mse = nn.MSELoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.scheduler_G = lr_sche.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.7, patience=10)
        self.scheduler_D = lr_sche.ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.7, patience=15)

        self.fs = nn.Sequential(
            nn.Linear(self.feats_in*2, self.feats_hidden),
            nn.ReLU(),
            nn.Linear(self.feats_hidden, self.feats_in)
        )
        self.sche = 0

        self.binary_pred = nn.Sigmoid()
        self.binary_real =  nn.Sigmoid()

    def adjust_lr(self, d_loss, g_loss):

        self.scheduler_D.step(d_loss)
        learn_rate_D = self.optimizer_D.param_groups[0]['lr']
        self.scheduler_G.step(g_loss)
        learn_rate_G = self.optimizer_G.param_groups[0]['lr']

        print(f'lr Dis: {learn_rate_D:}, lr Gen: {learn_rate_G}')

        return

    def forward(self, inputs, iftrain, id = 1, ep = 1):

        cond = inputs[0][:self.obs_length, :, :self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_in]#.permute(1, 0, 2).to(self.device)
        reals = inputs[0][:, :, :self.feats_in].unsqueeze(2).repeat(1, 1, self.K, 1)
        real_his = reals[:self.obs_length]

        if iftrain:
            fake_future = self.generator(cond)
            fakes = torch.concat([real_his, fake_future.detach()])
            fakes_grad = torch.concat([real_his, fake_future])

            self.optimizer_D.zero_grad()
            real_validity = self.discriminator(reals)
            fake_validity = self.discriminator(fakes)
            d_loss_real = self.loss_fn(real_validity, torch.ones_like(real_validity))
            d_loss_fake = self.loss_fn(fake_validity, torch.zeros_like(fake_validity))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()


            self.optimizer_G.zero_grad()
            fake_validity = self.discriminator(fakes_grad)
            g_loss = self.loss_fn(fake_validity, torch.ones_like(fake_validity))
            #mse_loss = self.loss_mse(fake_future[-1], tar_y[-1].unsqueeze(1).repeat(1, 20, 1))
            bi_pred = self.binary_pred(fake_future)
            bi_real = self.binary_real(tar_y).unsqueeze(2).repeat(1, 1, 20, 1)
            mse_loss = self.loss_fn(bi_pred, bi_real)
            g_loss = g_loss
            g_loss.backward()
            self.optimizer_G.step()

            if self.sche != ep:
                self.adjust_lr(d_loss, g_loss)
                self.sche = ep

        else:
            fake_future = self.generator(cond)
            g_loss = None

        if id % 100 == 0 :
            print(f"d_loss:{d_loss:.4f},g_loss{g_loss:.4f}")

        return fake_future, g_loss



