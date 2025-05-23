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
wandb.init(project="MONet for constellation")

#vis = visdom.Visdom()
#vis = visdom.Visdom(server='http://host.docker.internal', port=8097) #docker를 사용해서 생긴 문제(?)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpify(tensor):
    return tensor.cpu().detach().numpy()

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
            images, labelss = data
            images = images.to(device)
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
                masks_list_npy = [numpify(slot[:8]) for slot in output['masks_list']]
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]),
                                masks_list_npy)

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
    monet = model.Monet(conf,128, 128).to(device)
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
    monet = model.Monet(conf, 128, 128).to(device)
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
            images, labels = data
            images = images.to(device)

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


# #test할 때 visaulize해서 저장하는 코드
# def visualize_masks(images, masks, reconstructions, save_dir='./vis', prefix='test'):
#     os.makedirs(save_dir, exist_ok=True)
#     #for i in range(len(images)):
#     for i in range(8):
#         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#         axs[0].imshow(images[i].transpose(1, 2, 0))
#         axs[0].set_title('Input Image')
#         axs[1].imshow(masks[i].sum(axis=0))
#         axs[1].set_title('Mask')
#         axs[2].imshow(reconstructions[i].transpose(1, 2, 0))
#         axs[2].set_title('Reconstruction')
#         save_path = os.path.join(save_dir, f'{prefix}_{i}.png')
#         plt.savefig(save_path)
#         plt.close()
#         print(f"Saved visualization to {save_path}")

def visualize_masks(images, masks, reconstructions, masks_list=None, save_dir='./vis', prefix='test'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(8):
        # masks_list가 있으면 행 2, 없으면 행 1짜리
        num_mask_rows = 2 if masks_list is not None else 1
        fig, axs = plt.subplots(num_mask_rows, 9, figsize=(9,2))

        # 1행: 원본, 전체 마스크 합, 재구성
        axs[0, 0].imshow(images[i].transpose(1, 2, 0))
        axs[0, 0].set_title('Input')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(masks[i].sum(axis=0))
        axs[0, 1].set_title('Mask(Sum)')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(reconstructions[i].transpose(1, 2, 0))
        axs[0, 2].set_title('Reconstruction')
        axs[0, 2].axis('off')

        # 1행의 [3:] 칸은 흰색 화면으로
        for col in range(3, 9):
            axs[0, col].imshow(np.ones_like(images[i][0]), cmap='gray')  # H, W
            axs[0, col].set_title('')
            axs[0, col].axis('off')

        # 2행: masks_list가 있을 때만
        if masks_list is not None:
            # masks_list는 [num_slots][batch, C, H, W] 형태라 가정
            num_slots = len(masks_list)
            for slot_idx in range(num_slots):  # 최대 3개 슬롯만 표시 (3열) -> 없앳다 min(num_slots,3)이었는데 수정함ㅋ
                slot_mask = masks_list[slot_idx][i].sum(axis=0)  # 채널 축 합쳐서 (H,W)로
                axs[1, slot_idx].imshow(slot_mask)
                axs[1, slot_idx].set_title(f'Mask {slot_idx}')
                axs[1, slot_idx].axis('off')
            # 만약 slots가 3보다 적으면 빈 칸은 없애기
            for empty_idx in range(num_slots, 3):
                axs[1, empty_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{prefix}_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")


if __name__ == '__main__':
    sprite_experiment()
    #test하는 코드 추가!
    #sprite_experiment_test()
