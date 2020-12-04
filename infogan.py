import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import os


class FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(FrontEnd, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, x):
        # x: torch.Size([100, 1, 28, 28])
        print("here")
        output = self.main(x)
        return output


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        
        return output


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()

        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        output = self.main(x)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)

class Trainer:
    def __init__(self, G, FE, D, Q):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.batch_size = 100
        self.num_epoch = 100 

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(10, size=bs) # 隨機從數字0~9隨機給100個
        c = np.zeros((bs, 10)) # 準備紀錄C的特性
        c[range(bs),idx] = 1.0 # one-hot-encoder 上述idx
        dis_c.data.copy_(torch.Tensor(c)) # 複製上述 one-hot-encoder
        
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        # noise.shape: torch.Size([100, 62]) # noise 62 個維度
        # dis_c.shape: torch.Size([100, 10]) # 10個數字
        # con_c.shape: torch.Size([100, 2]) # 2個特性
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        # z: torch.Size([100, 74, 1, 1])
        return z, idx

    # =================================== Start to train ==========================================
    def train(self):
        # define torch.size
        real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()#　torch.Size([100, 1, 28, 28])
        label = torch.FloatTensor(self.batch_size, 1).cuda()# torch.Size([100, 1])
        dis_c = torch.FloatTensor(self.batch_size, 10).cuda()# 分類數字 # torch.Size([100, 10])
        con_c = torch.FloatTensor(self.batch_size, 2).cuda()# 分類特性 # # torch.Size([100, 2])
        noise = torch.FloatTensor(self.batch_size, 62).cuda()# # torch.Size([100, 62])
        
        # 轉成torch格式
        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)
        
        # define loss function
        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()
        
        # define optimal function(Adam)
        optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], 
                            lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}],
                            lr=0.001, betas=(0.5, 0.99))
        
        # load data
        dataset = dset.FashionMNIST('./dataset', train=True, transform=transforms.ToTensor(),  download=True)
        # dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True) # 60000
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1) # 600

        # fixed random variables
        # varying from -1 to 1(left to right).
        c = np.linspace(-1, 1, 10).reshape(1, -1) # (Row,Cloumn)(1, 10)
        c = np.repeat(c, 10, 0).reshape(-1, 1) # 重複10次 (100, 1)
        
        # (Width)
        c1 = np.hstack([c, np.zeros_like(c)]) # (100, 2) # 第一行與 c一樣 第二行全部是0
        # (Rotation)
        c2 = np.hstack([np.zeros_like(c), c]) # (100, 2) # 第一行全部是0 第二行與 c一樣
        
        idx = np.arange(10).repeat(10) # 0~9每個數字重複10次
        one_hot = np.zeros((100, 10)) # 定義一個都是0的陣列: (100, 10)
        one_hot[range(100), idx] = 1 # one-hot-encoder 上述idx
        
        fix_noise = torch.Tensor(100, 62).uniform_(-1, 1) #　因batch size = 100, noise = 62, torch.Size([100, 62])

        for epoch in range(self.num_epoch):
            for num_iters, batch_data in enumerate(dataloader, 0):
                # num_iters : 0~599
                # ========================= Train Discriminator ================================
                optimD.zero_grad()
                
                #　(real part)
                x, _ = batch_data # x:真實數據 _: label
                
                bs = x.size(0) # 100
                #         real_x.data.resize_(x.size())
                #         label.data.resize_(bs, 1)
                #         dis_c.data.resize_(bs, 10)
                #         con_c.data.resize_(bs, 2)
                #         noise.data.resize_(bs, 62)
                
                real_x.data.copy_(x) # 真實數據 torch.Size([100, 1, 28, 28])
                fe_out1 = self.FE(real_x) # torch.Size([100, 1024, 1, 1])
                probs_real = self.D(fe_out1) # torch.Size([100, 1]) # Discriminator 辨識真實數據的結果
                label.data.fill_(1) # label = 1 # torch.Size([100, 1])
                loss_real = criterionD(probs_real, label) # 算Discriminator辨識 與 真實label 的loss
                loss_real.backward() # 更新 算Discriminator辨識
                
                # (fake part)
                '''
                    dis_c 分類數字, con_c 分類特性, noise 隨機亂數, bs = 100
                    z: torch.Size([100, 74, 1, 1])
                    idx: 隨機從數字0~9隨機給100個
                    noise.shape: torch.Size([100, 62]) # noise 62 個維度
                    dis_c.shape: torch.Size([100, 10]) # 10個數字
                    con_c.shape: torch.Size([100, 2]) # 2個特性
                '''
                z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z) # generate fake image
                '''
                    當我們在訓練網絡的時候可能希望保持一部分的網絡參數不變，
                    只對其中一部分的參數進行調整；或者值訓練部分分支網絡，
                    並不讓其梯度對主網絡的梯度造成影響，
                    這時候我們就需要使用detach()函數來切斷一些分支的反向傳播
                    '''
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()
                
                # ========================= Train G ================================
                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)# fake_x: generate fake image
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)
                
                # ========================= Train Q ================================
                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
                
                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()
            
                if num_iters % 100 == 0 or num_iters == 599:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                                epoch, num_iters, D_loss.data.cpu().numpy(),
                                G_loss.data.cpu().numpy()))          
                    noise.data.copy_(fix_noise) # 上述給定的隨機亂數
                    dis_c.data.copy_(torch.Tensor(one_hot)) #　上述給定的欲建構的數字

                    con_c.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1) # 62+10+2 #torch.Size([100, 74, 1, 1])
                    x_save = self.G(z) # generate image
                    save_image(x_save.data, './out2/c1_%s_%s.png'%(epoch,num_iters), nrow=10)

                    con_c.data.copy_(torch.from_numpy(c2))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                    x_save = self.G(z) # generate image
                    save_image(x_save.data, './out2/c2_%s_%s.png'%(epoch,num_iters), nrow=10)

if __name__ == "__main__": 
    fe = FrontEnd() # discriminator and Q

    d = D().cuda()
    q = Q().cuda()
    g = G().cuda()

    print("================================ FrontEnd =================================")
    print(fe)
    print("================================ D =================================")
    print(d)
    print("================================ Q =================================")
    print(q)
    print("================================ G =================================")
    print(g)


    # 初始化參數
    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)
        print(type(i))
            
        trainer = Trainer(g, fe, d, q)
        trainer.train()