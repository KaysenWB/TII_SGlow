import pandas as pd

from dataloader import *
from utils import *
from dataloader import Dataset_ship
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from Models.Compared_models import *
from Models.STGCN import ST_GCN
from Models.VAE import VAE_tra
from Models.CVAE import CVAE_tra
from Models.GAN import GAN_tra
from Models.Glow import Glow_tra
from Models.RealNVP import RealNVP_tra
from Models.Ablas import *
from Models.SGlow import SGlow_tra
from torchprofile import profile_macs
from thop import profile


class processor(object):
    def __init__(self, args):

        self.args = args
        self.device = self.args.device

        model_dict = {'Trans': Trans,
                      'LSTM': LSTM,
                      'GRU': GRU,
                      'Seq2Seq': Seq2Seq,
                      'TCNN': T_CNN,
                      'STGCN':ST_GCN,
                      'VAE': VAE_tra,
                      'CVAE':CVAE_tra,
                      'GAN': GAN_tra,
                      'RealNVP': RealNVP_tra,
                      'Glow':Glow_tra,
                      'SGlow':SGlow_tra,

                      'aff_conv_D':aff_conv_D_tra,
                      'E_aff_D': E_aff_D_tra,
                      'E_aff_conv':E_aff_conv_tra,
                      'E_D':E_D_tra,
                      'aff_conv':aff_conv_tra,
                      'E_conv_D':E_conv_D_tra,
                      'D_': D_tra,
                      'E_': E_tra,

        }

        self.net = model_dict[self.args.train_model](args)
        self.net.to(self.device)

        self.set_optimizer()
        self.dataset = Dataset_ship(args)
        self.train_data_len = self.dataset.get_length(flag='train')
        self.test_data_len = self.dataset.get_length(flag='test')
        self.feats_out = self.args.feats_out

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1


    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        self.model_save_path = self.args.save_dir + '/'  + self.args.train_model + '_' + \
                                    str('best') + '.tar'
        print(self.model_save_path)
        if os.path.isfile(self.model_save_path):
            print('Loading model')
            checkpoint = torch.load(self.model_save_path)
            self.net.load_state_dict(checkpoint['state_dict'])


    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.7, patience=6)

        if self.args.train_model == 'Glow' :
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=3)

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        st = time.time()
        ADE, FDE , ME, pa= self.test_epoch()
        print('Test_ADE: {} Test_FDE: {} MAE_sog: {} MSE_sog: {} MAE_cog: {} MSE_cog: {}'
              .format( ADE, FDE, ME[0], ME[1], ME[2], ME[3]))
        return  ADE, FDE, ME, time.time() - st, pa

    def train(self):

        print('Training begin')
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)

            self.net.eval()
            ade, fde, ME , _= self.test_epoch()
            if ade < self.best_ade:
                self.best_ade = ade
                self.best_epoch = epoch
                self.best_fde = fde
                self.save_model('best')

            print('----epoch {}, train_loss={:.5f}, ADE ={:.5f}, FDE ={:.5f}, Best_ADE={:.5f}, Best_FDE={:.5f} '
                .format(epoch, train_loss, ade, fde , self.best_ade, self.best_fde))

    def train_epoch(self, epoch):

        self.dataset.pointer(flag='train')
        loss_epoch = 0
        custom_collate_fn = lambda batch: batch[0]
        train_loader = DataLoader(dataset=self.dataset, shuffle=True, collate_fn=custom_collate_fn)

        for b_id, batch in enumerate(train_loader):
            start = time.time()

            inputs = self.dataset.batch_operation(batch)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            if self.args.train_model != 'GAN':
                _, loss = self.net.forward(inputs, iftrain=True)
                self.optimizer.zero_grad()
                loss_epoch += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
                self.optimizer.step()
            else:
                _, loss = self.net.forward(inputs, iftrain=True, id=b_id, ep=epoch)

            end = time.time()
            if b_id % 100 == 0:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                b_id,self.train_data_len, epoch, loss.item(), end - start))

        if self.args.train_model != 'GAN':
            self.scheduler.step(loss)
            learn_rate = self.optimizer.param_groups[0]['lr']
            print(f'Learn Rate: {learn_rate}')
        train_loss_epoch = loss_epoch / self.train_data_len
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self, if_sc = False):

        self.dataset.pointer(flag='test')
        custom_collate_fn = lambda batch: batch[0]
        test_loader = DataLoader(dataset=self.dataset, shuffle=False, collate_fn=custom_collate_fn)
        ADE, FDE, ME = [], [], []

        for b_id, batch in enumerate(test_loader):

            inputs = self.dataset.batch_operation(batch)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            pred_tra, _ = self.net.forward(inputs, iftrain=False)
            pred_traj = pred_tra.detach().cpu().numpy()
            tar_y = inputs[0][self.args.obs_length:, :, :self.feats_out].detach().cpu().numpy()
            ade, fde, Mes = Metrics(pred_traj, tar_y)
            ADE.append (ade), FDE.append (fde), ME.append(Mes)
            self.net.zero_grad()

        errorp = pd.DataFrame()
        errorp['ADE_dis'] = ADE
        errorp['FDE_dis'] = FDE
        np.save(self.args.save_dir + '/Error.npy', errorp.values)

        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return np.mean(ADE), np.mean(FDE), np.array(ME).mean(0), num_params,

    @torch.no_grad()
    def pred(self):

        print('Preding begin')
        self.dataset.pointer(flag='pred')
        self.load_model()
        self.net.eval()

        custom_collate_fn = lambda batch: batch[0]
        pred_loader = DataLoader(dataset=self.dataset, shuffle=False, collate_fn=custom_collate_fn)

        for b_id, batch in enumerate(pred_loader):

            inputs, real_tra = self.dataset.batch_operation(batch, ifpred=True)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            pred_tra, _ = self.net.forward(inputs, iftrain=False)
            pred_tra = pred_tra.detach().cpu().numpy()

            np.save(self.args.save_dir + '/Preds.npy', pred_tra)
            np.save(self.args.save_dir + '/Reals.npy', real_tra)
            print('Preds saved')

        self.net.zero_grad()

        return