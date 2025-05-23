# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists

import torchvision


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        ys, xs = torch.meshgrid(ys, xs)
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
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        # masks 리스트를 그대로 tensor로 concat하기 전 상태로 저장
        masks_list = masks.copy()        

        masks = torch.cat(masks, 1)
        tr_masks = masks.permute(0, 2, 3, 1)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,           # 합쳐진 마스크 (B, K, H, W)
                'masks_list': masks_list, # 합치기 전 리스트 (각 요소: (B,1,H,W))
                'reconstructions': full_reconstruction}


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


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


