# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
import torchvision

from torch_geometric.nn import GCNConv
from scipy.optimize import linear_sum_assignment
import itertools

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#MONet
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
    #convolution + batch_normalization + ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size,stride=stride,padding=padding,
                            bias=True)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbr=nn.Sequential(*layers)
            return cbr
        
        #contracting path
        self.enc1_1=CBR2d(in_channels=4,out_channels=64) #원래 in_channels=1 #4=3channels+1scope
        self.enc1_2=CBR2d(in_channels=64,out_channels=64)

        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.enc2_1=CBR2d(in_channels=64, out_channels=128)
        self.enc2_2=CBR2d(in_channels=128,out_channels=128)

        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.enc3_1=CBR2d(in_channels=128, out_channels=256)
        self.enc3_2=CBR2d(in_channels=256,out_channels=256)

        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.enc4_1=CBR2d(in_channels=256, out_channels=512)
        self.enc4_2=CBR2d(in_channels=512,out_channels=512)

        self.pool4=nn.MaxPool2d(kernel_size=2)

        self.enc5_1=CBR2d(in_channels=512,out_channels=1024)

        #Expansive path
        self.dec5_1=CBR2d(in_channels=1024,out_channels=512)

        self.unpool4=nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2=CBR2d(in_channels=2*512,out_channels=512) #in_channels가 두배인 이유는 encoder의 일부가 붙기때문(skip connection)
        self.dec4_1=CBR2d(in_channels=512,out_channels=256)
    
        self.unpool3=nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2=CBR2d(in_channels=2*256,out_channels=256) 
        self.dec3_1=CBR2d(in_channels=256,out_channels=128)

        self.unpool2=nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2=CBR2d(in_channels=2*128,out_channels=128) 
        self.dec2_1=CBR2d(in_channels=128,out_channels=64)

        self.unpool1=nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2=CBR2d(in_channels=2*64,out_channels=64) 
        self.dec1_1=CBR2d(in_channels=64,out_channels=64)
        
        self.fc=nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1,stride=1,padding=0,bias=True) #원래 out_channels=1

    def forward(self,x):
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)

        enc2_1=self.enc2_1(pool1)
        enc2_2=self.enc2_2(enc2_1)
        pool2=self.pool2(enc2_2)

        enc3_1=self.enc3_1(pool2)
        enc3_2=self.enc3_2(enc3_1)
        pool3=self.pool3(enc3_2)

        enc4_1=self.enc4_1(pool3)
        enc4_2=self.enc4_2(enc4_1)
        pool4=self.pool4(enc4_2)

        enc5_1=self.enc5_1(pool4)

        dec5_1=self.dec5_1(enc5_1)

        unpool4=self.unpool4(dec5_1)
        cat4=torch.cat([unpool4, enc4_2], dim=1) #dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2=self.dec4_2(cat4)
        dec4_1=self.dec4_1(dec4_2)

        unpool3=self.unpool3(dec4_1)
        cat3=torch.cat([unpool3, enc3_2],dim=1)
        dec3_2=self.dec3_2(cat3)
        dec3_1=self.dec3_1(dec3_2)

        unpool2=self.unpool2(dec3_1)
        cat2=torch.cat([unpool2,enc2_2], dim=1)
        dec2_2=self.dec2_2(cat2)
        dec2_1=self.dec2_1(dec2_2)

        unpool1=self.unpool1(dec2_1)
        cat1=torch.cat([unpool1, enc1_2], dim=1)
        dec1_2=self.dec1_2(cat1)
        dec1_1=self.dec1_1(dec1_2)

        x=self.fc(dec1_1)
        
        return x

class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet().to(device)
                        # (num_blocks=conf.num_blocks,
                        #  in_channels=4,
                        #  out_channels=2,
                        #  channel_base=conf.channel_base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4): 
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 32)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(18, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs, indexing='xy')
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result

class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.beta = conf.beta
        self.gamma = conf.gamma

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        zs=[] #...............latent vector 모음 (constellation을 위해 추가)
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            zs.append(z) #............latent vector 모음 (constellation을 위해 추가)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        zs=torch.stack(zs,1) #...........latent vector 모음 차원 합치기

        # masks 리스트를 그대로 tensor로 concat하기 전 상태로 저장
        masks_list = masks.copy()

        masks = torch.cat(masks, 1)
        tr_masks = masks.permute(0, 2, 3, 1)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])

        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,           # 합쳐진 마스크 (B, K, H, W)
                'masks_list': masks_list, # 합치기 전 리스트 (각 요소: (B,1,H,W))
                'reconstructions': full_reconstruction,
                'zs': zs #........latent vector 모음
                }


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :16]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 16:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred


#필요없을거같은데 지워,,
# def print_image_stats(images, name):
#     print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
#     print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
#     print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())



#SCAN (train시키는 코드 필요)
class BVAE(nn.Module): #학습 시 scan의 이미지 처리 단(beta VAE)
    def __init__(self, input_size, hidden_size, output_size):
        super(BVAE, self).__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2)  # 인코더가 뽑아야 할 output이 두개 (mu(평균), logvar(분산))
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu, logvar = encoder_output[:, :self.output_size], encoder_output[:, self.output_size:]
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) #sigma=root(var)=root(exp(log(var))) ->양수로 만들기 위해서 =exp(0.5*log(var))
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def compute_loss(self, x, recon_x, mu, logvar, dae, beta):
        z_d = dae.encode(x)
        recon_z_d = dae.encode(recon_x)
        delta = z_d - recon_z_d
        L2_loss = 0.5 * torch.sum(delta * delta)
        KLD_loss = -0.5 * beta * torch.sum(1 + logvar - mu * mu - logvar.exp())
        return L2_loss, KLD_loss

class DAE(nn.Module): #denoising autoencoder, noise를 제거하는 VAE로 Beta VAE 학습 과정에서 오차 보정을 위해 사용.
    def __init__(self, input_size, hidden_size, output_size):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False)
        )

        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(output_size, hidden_size)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

        self.elu = nn.ELU(inplace=False)
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.encoder(x)
        z = self.tanh(self.fc1(h))
        return z

    def decode(self, z):
        h = self.elu(self.fc2(z))
        out = self.decoder(h)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def compute_loss(self, x_org, x_out):
        delta = x_org - x_out
        reconstr_loss = 0.5 * torch.sum(delta * delta)
        return reconstr_loss

class SCAN(nn.Module): #학습시 라벨 입력 단이자 scan 모델에서 최종적으로 학습시켜야 하는 것. 나중에 추론 및 생성에서 이 부분 이용.
    def __init__(self, input_size, hidden_size, output_size):
        super(SCAN, self).__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2)  # 인코더가 뽑아야 할 output이 두개
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, y):
        encoder_output = self.encoder(y)
        mu, logvar = encoder_output[:, :self.output_size], encoder_output[:, self.output_size:]
        logvar = torch.clamp(logvar, min=-10, max=10) #NaN값 나오는 이슈 방지
        z = mu + torch.randn_like(torch.exp(0.5 * logvar))
        out_y = self.decoder(z)
        return z, out_y, mu, logvar

    def compute_loss(self, x, y, target, mu, logvar, x_mu, x_logvar, beta=10.0 ,lambd=1.0): #...beta 없었고 lambd=10이었는데 논문대로 고쳐봄
        # y, mu, logvar: image
        # x_mu, x_logvar: label
        reconstr_loss = nn.BCELoss()(y, target) #loss function 바꿔봄
        KLD_loss_1 = -beta * 0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())
        KLD_loss_2 = -lambd * self._kl(x_mu, x_logvar, mu, logvar) #.................beta는 설정안하나????
        return reconstr_loss, KLD_loss_1, KLD_loss_2

    def _kl(self, mu1, logvar1, mu2, logvar2):
        mu = mu1 - mu2
        return torch.sum(0.5 * (logvar2 - logvar1 + (logvar1.exp() - logvar2.exp()) + mu * mu / logvar2.exp() - 1))

class Recombinator(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(Recombinator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv1d(4, 1024, 1), #mu0, logvar0, mu1, logvar1
            nn.ReLU(),
            nn.Conv1d(1024, 6, 1) #.........앞에 3개평균내서 r_mu, 뒤에 3개 평균내서 r_logvar로 쓰던데 왜 3개씩일까
        )

    def recombine(self, mu0, logvar0, mu1, logvar1):
        z_stacked = torch.stack([mu0, mu1, logvar0, logvar1], 1)
        return z_stacked

    def reparameterize(self, mu, logvar): #주어진 평균과 분산을 이용하여 정규분포에서 샘플링된 latent 벡터 z 생성
        std = logvar.mul(0.5).exp_()
        #eps = Variable(std.data.new(std.size()).normal_())
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, mu0, logvar0, mu1, logvar1):
        z_stacked = self.recombine(mu0, logvar0, mu1, logvar1)
        h = self.conv(z_stacked)
        mu, logvar = torch.split(h, 3, 1)
        r_mu = torch.mean(mu, 1)  # 평균 사용으로 결합
        r_logvar = torch.mean(logvar, 1)  # 평균 사용으로 결합
        r_z = self.reparameterize(r_mu, r_logvar)
        return r_z, r_mu, r_logvar

    def compute_loss(self, r_mu, r_logvar, x_mu, x_logvar, y_mu, y_logvar):
        symbol_loss = self._kl(y_mu, y_logvar, r_mu, r_logvar) #...forward kl 아마도.,?
        return _, symbol_loss

    def _kl(self, mu1, logvar1, mu2, logvar2):
        mu = mu1 - mu2
        return torch.sum(0.5 * (logvar2 - logvar1 + (logvar1 - logvar2).exp() + torch.mul(mu, mu) / logvar2.exp() - 1))


#GNN
class RelationalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, r_dim, num_slots):
        super(RelationalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, r_dim)
        self.fc_logvar = nn.Linear(hidden_dim, r_dim)
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.global_mlp = nn.Sequential(
            nn.Linear(num_slots * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        ).to(device)


    def forward(self, x, edge_index):
        batch_size, num_slots, features = x.shape

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = x.view(batch_size, num_slots, self.hidden_dim)

        x = x.view(batch_size, -1)
        global_info = self.global_mlp(x)

        mu_q = self.fc_mu(global_info)
        logvar_q = self.fc_logvar(global_info)
        return mu_q, logvar_q

# LSTM
class SequentialLSTM(nn.Module):
    def __init__(self, r_dim, hidden_dim, latent_dim, num_slots):
        super(SequentialLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(r_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mu 추출
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # logvar 추출
        self.num_slots = num_slots

    def forward(self, r): #r: representation
        batch_size, r_dim = r.size()
        h_t, c_t = self.init_hidden(batch_size, r.device)

        mu_outputs = []
        logvar_outputs = []
        for i in range(self.num_slots):
            h_t, c_t = self.lstm_cell(r, (h_t, c_t))  # 각 슬롯에 대해 LSTM 처리
            mu_i = self.fc_mu(h_t)  # mu 추출
            logvar_i = self.fc_logvar(h_t)  # logvar 추출
            mu_outputs.append(mu_i)
            logvar_outputs.append(logvar_i)

        # [batch_size, num_slots, latent_dim] 형태로 출력 쌓기
        mu_outputs = torch.stack(mu_outputs, dim=1)
        logvar_outputs = torch.stack(logvar_outputs, dim=1)
        return mu_outputs, logvar_outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_hidden(self, batch_size,device):
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        return h, c


#Constellation
class Constellation(nn.Module):
    def __init__(self, conf, monet, input_dim, hidden_dim, output_dim, latent_dim, r_dim, height, width):
        super(Constellation, self).__init__()
        self.monet = monet
        self.gnn = RelationalGNN(latent_dim, hidden_dim, r_dim, conf.num_slots)
        self.lstm = SequentialLSTM(r_dim, hidden_dim, latent_dim, conf.num_slots)
        # gpt는 추가 loss function 없이 알아서 위치 추출을 하도록 학습된다고 주장하지만 확인 필요할 것으로 보임
        self.mask_extractor = nn.Sequential(
            nn.Conv1d(conf.num_slots, 16, 3, padding=1), #차원 이슈 해결
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.Softmax(dim=2)
        )    
    
    def encode(self, x):
        monet_output = self.monet(x)
        masks = monet_output['masks']
        recon = monet_output['reconstructions']
        o = monet_output['zs']
        batch_size = o.shape[0]
        learned_mask = self.mask_extractor(o) #.............monet_output의 latent vector만 가지고 mask extractor가 된다고,,?

        a = o * learned_mask

        residue = o * (1 - learned_mask)
        num_nodes = o.shape[1]
        edge_index = self.create_fully_connected_edge_index(num_nodes)
        edge_index = edge_index.to(device)
        mu_q, logvar_q = self.gnn(a, edge_index)
        r = self.lstm.reparameterize(mu_q, logvar_q)
        return r, mu_q, logvar_q, a, o, learned_mask, recon, residue

    def decode(self, r):
        mu_outputs, logvar_outputs = self.lstm(r)
        recon = self.lstm.reparameterize(mu_outputs, logvar_outputs)
        return recon, mu_outputs, logvar_outputs

    #완전 연결 edge index 생성 함수.
    def create_fully_connected_edge_index(self, num_nodes):
        edge_index = list(itertools.permutations(range(num_nodes), 2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

class LossFunctions(nn.Module):
    def __init__(self, conf, beta=4, gamma_init=0.1, latent_dim=16):
        super(LossFunctions, self).__init__()
        self.beta = beta
        self.gamma = nn.Parameter(torch.tensor([gamma_init] * conf.num_slots * latent_dim))

    # 논문에 나온 손실 함수 구현
    def reconstruction_loss(self, ai, a_hat):
        loss = 0.0
        batch_size, num_slots, latent_dim = ai.size()

        for b in range(batch_size):
            # 각 배치에 대해 cost_matrix를 생성
            cost_matrix = np.zeros((num_slots, num_slots))
            for i in range(num_slots):
                for j in range(num_slots):
                    cost_matrix[i, j] = np.linalg.norm(ai[b, i].detach().cpu().numpy() - a_hat[b, j].detach().cpu().numpy())

            # Hungarian matching (linear_sum_assignment) 적용
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 매칭된 슬롯 쌍에 대해 재구성 손실 계산
            for i, j in zip(row_ind, col_ind):
                loss += 0.5 * torch.norm(ai[b, i] - a_hat[b, j]) ** 2

        return loss

    def kl_divergence(self, mu_q, logvar_q, mu_p=None, logvar_p=None):
        if mu_p is None or logvar_p is None:
            mu_p = torch.zeros_like(mu_q)
            logvar_p = torch.zeros_like(logvar_q)
        kld = -0.5 * torch.sum(1 + logvar_q - logvar_p - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        return kld

    def mask_entropy_loss(self, learned_mask):
        # 배치와 슬롯 차원을 모두 포함하여 손실 계산
        loss = -torch.sum(learned_mask * torch.log(learned_mask + 1e-10))
        return loss

    def conditioning_loss(self, o, a_hat, learned_mask):
        loss = 0.0
        batch_size, num_slots, latent_dim = o.size()

        # learned_mask의 차원을 num_slots에 맞춰서 반복 (batch_size, num_slots, latent_dim)
        learned_mask = learned_mask.repeat(1, num_slots, 1)

        for b in range(batch_size):
            cost_matrix = np.zeros((num_slots, num_slots))

            # 각 배치 내에서 o와 a_hat 간의 cost_matrix 계산
            for i in range(num_slots):
                for j in range(num_slots):
                    cost_matrix[i, j] = np.linalg.norm(
                        o[b, i].detach().cpu().numpy() - a_hat[b, j].detach().cpu().numpy() / learned_mask[b, j].detach().cpu().numpy()
                    )

            # Hungarian matching (linear_sum_assignment) 적용
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 매칭된 쌍에 대해 손실 계산
            for i, j in zip(row_ind, col_ind):
                l_rec_star = 0.5 * torch.sum((o[b, i] - a_hat[b, j] / learned_mask[b, j]) ** 2)
                gamma_j = self.gamma[j]  # 학습 가능한 gamma 변수
                loss += torch.sum((1 - learned_mask[b, j]) * torch.abs(l_rec_star - gamma_j))

        return loss

    def reordering_loss(self, a_hat):
        loss = 0.0
        batch_size, num_slots, latent_dim = a_hat.size()

        # 각 배치 내에서 인접한 슬롯들의 차이를 계산
        for b in range(batch_size):
            for i in range(1, num_slots):
                loss += torch.norm(a_hat[b, i] - a_hat[b, i - 1]) ** 2

        return loss

    def total_loss(self, ai, a_hat, mu_q, logvar_q, o, learned_mask):
        L_rec = self.reconstruction_loss(ai, a_hat)
        L_kl = self.beta * self.kl_divergence(mu_q, logvar_q)
        L_entropy = self.mask_entropy_loss(learned_mask)
        L_condition = self.conditioning_loss(o, a_hat, learned_mask)
        L_reorder = self.reordering_loss(a_hat)

        L_total = L_rec + L_reorder + L_kl + L_entropy + L_condition
        return L_total