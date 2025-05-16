# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import os

import model
import datasets
import config

import matplotlib.pyplot as plt

import wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="MONet")

#vis = visdom.Visdom()
#vis = visdom.Visdom(server='http://host.docker.internal', port=8097) #docker를 사용해서 생긴 문제(?)



def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    #vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])

def run_training(monet, conf, trainloader):
    os.makedirs(os.path.dirname(conf.checkpoint_file), exist_ok=True) #여기랑 다음줄까지,,,,,,,,ㅜㅜ
    checkpoint_dir = './checkpoints'
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(conf.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % conf.vis_every == conf.vis_every-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / conf.vis_every))
                wandb.log({"Loss/train_monet": running_loss / conf.vis_every, "epoch": epoch}) #wandb 기록      
                running_loss = 0.0
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]))

        torch.save(monet.state_dict(), conf.checkpoint_file)

    print('training done')

def sprite_experiment():
    conf = config.sprite_config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    #데이터셋 없으면 생성하는 코드
    trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform) 
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet(conf, 64, 64).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

def clevr_experiment():
    conf = config.clevr_config
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr(conf.data_dir,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    monet = model.Monet(conf, 128, 128).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

#test하는 코드
def sprite_experiment_test():
    conf = config.sprite_config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    #데이터셋 없으면 생성하는 코드
    trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform) 
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet(conf, 64, 64).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_testing(monet, conf, trainloader)

#test하는 코드 #나중에 testloader만들어서 trainloader 다 대체해야돼!!
def run_testing(monet, conf, trainloader):
    # 모델 파라미터 로드
    if os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        print('No checkpoint found at', conf.checkpoint_file)
        return

    monet.eval()  # 평가 모드
    total_loss = 0.0

    with torch.no_grad():  # 평가 시에는 gradient 필요 없음
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            images = images.cuda()

            output = monet(images)
            loss = torch.mean(output['loss'])
            total_loss += loss.item()

            # 특정 주기마다 결과 시각화
            if i % conf.vis_every == conf.vis_every - 1:
                print('[Test %5d] loss: %.3f' % (i + 1, total_loss / (i + 1)))
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]))

    avg_loss = total_loss / len(trainloader)
    print('Test done. Average loss:', avg_loss)
    wandb.log({"Loss/test_monet": avg_loss})


#test할 때 visaulize해서 저장하는 코드
def visualize_masks(images, masks, reconstructions, save_dir='./vis', prefix='test'):
    os.makedirs(save_dir, exist_ok=True)
    #for i in range(len(images)):
    for i in range(8):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(images[i].transpose(1, 2, 0))
        axs[0].set_title('Input Image')
        axs[1].imshow(masks[i].sum(axis=0))
        axs[1].set_title('Mask')
        axs[2].imshow(reconstructions[i].transpose(1, 2, 0))
        axs[2].set_title('Reconstruction')
        save_path = os.path.join(save_dir, f'{prefix}_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")



if __name__ == '__main__':
    #clevr_experiment()
    sprite_experiment()
    #test하는 코드 추가!
    sprite_experiment_test()
