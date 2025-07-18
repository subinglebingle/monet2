import random
import config

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom
import os

#import models
#import datasets
from models import Constellation, SCAN, Monet, BVAE, DAE, Recombinator
import config
from datasets import Sprites

import matplotlib.pyplot as plt

#import wandb
#wandb.login(key=os.getenv("WANDB_API_KEY"))
#wandb.init(project="Constellation test")

# Device 설정
device = torch.device('cpu')  # CPU에서 실행하도록 설정

# TensorBoard writer 초기화
# writer = SummaryWriter(log_dir='./runs/eval_logs')

# 모델 파라미터 설정
input_dim = 262144  # 차원 맞추기(16*128*128)
hidden_dim = 128
output_dim = 64
latent_dim = 16
r_dim = 16
height = 128  # 입력 이미지의 높이
width = 128   # 입력 이미지의 너비
batch_size = 1  # 평가 시에는 하나의 이미지에 대해 처리

# Config 설정
conf = config.sprite_config

def process_labels(labels, word_to_idx, num_classes=379):#여러 텍스트 라벨을 받아 멀티-핫 인코딩으로 변환하는 함수
    # 멀티핫 인코딩을 위한 빈 벡터 생성
    one_hot_labels = torch.zeros(1, num_classes, dtype=torch.float)  # 배치 크기는 1 (평가 시 하나의 입력만 사용)

    for label in labels:
        # 각 텍스트 라벨이 word_to_idx 사전에 있는지 확인
        if label in word_to_idx:
            index = word_to_idx[label]  # 사전에서 해당 라벨의 인덱스를 가져옴
            one_hot_labels[0, index] = 1  # 해당 인덱스에 1 설정
        else:
            print(f"Warning: Label '{label}' not found in word_to_idx. Skipping.")

    return one_hot_labels

word_to_idx = {
    # x_pos 라벨 (수평 위치)
    "left": 0,  # 왼쪽 (Left)
    "left-centre": 1,  # 왼쪽 중앙 (Left-centre)
    "centre": 2,  # 중앙 (Centre)
    "right-centre": 3,  # 오른쪽 중앙 (Right-centre)
    "right": 4,  # 오른쪽 (Right)

    # y_pos 라벨 (수직 위치)
    "bottom": 5,  # 아래 (Bottom)
    "bottom-middle": 6,  # 아래 중앙 (Bottom-middle)
    "middle": 7,  # 중앙 (Middle)
    "top-middle": 8,  # 위쪽 중앙 (Top-middle)
    "top": 9,  # 위쪽 (Top)

    # num_sprites 라벨 (스프라이트 개수)
    "4_sprites": 10,  # 4개의 스프라이트
    "5_sprites": 11,  # 5개의 스프라이트
    "6_sprites": 12,  # 6개의 스프라이트
    "7_sprites": 13,  # 7개의 스프라이트

    # curviness 라벨 (곡률)
    "straight": 14,  # 직선 (Straight)
    "bend": 15,  # 굽은 선 (Bend)
    "arch": 16,  # 아치형 (Arch)
    "horseshoe": 17,  # 말굽형 (Horseshoe)
    "circle": 18,  # 원형 (Circle)

    # orientation 라벨 (회전 각도)
}

# orientation (회전 각도)의 경우 그대로 0 ~ 360 범위 내에서 값을 사용
for angle in range(0, 360):
    word_to_idx[str(angle)] = angle + 19

num_classes = len(word_to_idx) #379

#없앨지말지,,,, 왜있음 이거
def process_labels1(batch_labels, num_classes): #트레이닝과 데이터셋 생성시 이용한 라벨링 방식과 텍스트 인풋이 달라서 부득이하게 추가 ??????
#def process_labels1(batch_labels, word_to_idx, num_classes):
    batch_size = batch_labels.size(0)  # 배치 크기 추출
    num_labels = batch_labels.size(1)  # 각 배치의 라벨 개수 추출

    indices = torch.zeros(batch_size, num_labels, dtype=torch.long, device=batch_labels.device)

    # # 각 배치의 라벨을 인덱스로 변환
    # for i in range(batch_size):
    #     for j in range(num_labels):
    #         label = int(batch_labels[i, j].item())  # 라벨 값을 정수형으로 변환
    #         if label in word_to_idx:
    #             indices[i, j] = word_to_idx[label]  # word_to_idx에서 인덱스 가져오기
    #         else:
    #             print(f"Warning: Label {label} not found in word_to_idx.")
    #             indices[i, j] = 0  # 기본값으로 0 추가

    # 멀티핫 인코딩을 위한 빈 벡터 생성 (배치마다 생성)
    one_hot_labels = torch.zeros(batch_size, num_classes, device=indices.device, dtype=torch.float)

    # scatter를 통해 배치별로 멀티핫 인코딩 수행
    one_hot_labels.scatter_(1, indices, 1)

    return one_hot_labels

# word_to_idx1 = {  
#     # x_pos 라벨 (수평 위치)
#     0: -1.0, #x_pos, y_pos 계산 중에 0으로 나와야 하는 라벨이 -1이 되는 가능성이 존재함. 이미 학습된 체크포인트를 어찌 할 수 없어 조치함
#     0: 0.0,  # 왼쪽 (Left)
#     1: 1.0,  # 왼쪽 중앙 (Left-centre)
#     2: 2.0,  # 중앙 (Centre)
#     3: 3.0,  # 오른쪽 중앙 (Right-centre)
#     4: 4.0,  # 오른쪽 (Right)

#     # y_pos 라벨 (수직 위치)
#     5: -1.0, #역시 비슷한 사유
#     5: 0.0,  # 아래 (Bottom)
#     6: 1.0,  # 아래 중앙 (Bottom-middle)
#     7: 2.0,  # 중앙 (Middle)
#     8: 3.0,  # 위쪽 중앙 (Top-middle)
#     9: 4.0,  # 위쪽 (Top)

#     # num_sprites 라벨 (스프라이트 개수)
#     10: 4.0,  # 4개의 스프라이트
#     11: 5.0,  # 5개의 스프라이트
#     12: 6.0,  # 6개의 스프라이트
#     13: 7.0,  # 7개의 스프라이트

#     # curviness 라벨 (곡률)
#     14: 0.0,  # 직선 (Straight)
#     15: 1.0,  # 굽은 선 (Bend)
#     16: 2.0,  # 아치형 (Arch)
#     17: 3.0,  # 말굽형 (Horseshoe)
#     18: 4.0,  # 원형 (Circle)

#     # orientation 라벨 (회전 각도)
#     # 각도 값은 0 ~ 360 범위 내에서 그대로 사용
# }

# for angle in range(0, 360):
#     word_to_idx1[angle+19] = float(angle)  # 곡률 라벨은 19~379까지 그대로 사용


# num_classes1 = len(word_to_idx1) #5+5+4+5+360=379
num_classes1=379

# Transform 적용
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset 불러오기
dataset = Sprites(conf.data_dir, n=100000, canvas_size=128, train=False, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화
monet = Monet(conf, height, width).to(device)
dae = DAE(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)
bvae = BVAE(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)
scan = SCAN(input_size=num_classes, hidden_size=hidden_dim, output_size=r_dim).to(device)
model = Constellation(conf, monet, latent_dim, hidden_dim, output_dim, latent_dim, r_dim, height, width).to(device) #input 차원 맞추기
recomb = Recombinator(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)

# DataParallel에서 저장된 모델 체크포인트 로드 시 'module.' 접두사 제거 함수
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # 'module.' 접두사 제거
        new_state_dict[new_key] = value
    return new_state_dict

# 모델의 학습된 파라미터 불러오기 (DataParallel을 제거하여 로드)
checkpoint = torch.load('combined_models.pt', map_location=device)

# 체크포인트에서 'module.' 접두사 제거 후 파라미터 로드
monet.load_state_dict(remove_module_prefix(torch.load(conf.checkpoint_file)))
model.load_state_dict(remove_module_prefix(torch.load(os.path.join(conf.checkpoint_dir, 'constellation.ckpt'))))
scan.load_state_dict(remove_module_prefix(torch.load(os.path.join(conf.checkpoint_dir, 'scan.ckpt'))))
bvae.load_state_dict(remove_module_prefix(torch.load(os.path.join(conf.checkpoint_dir, 'bvae.ckpt'))))
dae.load_state_dict(remove_module_prefix(torch.load(os.path.join(conf.checkpoint_dir, 'dae.ckpt'))))
recomb.load_state_dict(remove_module_prefix(torch.load(os.path.join(conf.checkpoint_dir, 'recomb.ckpt'))))

# 모델을 평가 모드로 전환
model.eval()
monet.eval()
scan.eval()
bvae.eval()
dae.eval()
recomb.eval()

test_num=input("몇 개의 이미지를 평가하시겠습니까?: ")

# 임의의 이미지를 선택하여 평가
for idx, (image, label) in enumerate(data_loader):
    if idx < test_num:  # 첫 번째 이미지를 사용
        image = image.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)

        print("원본 이미지:")
        print(f"Label: {label}")

        original_image = transforms.ToPILImage()(image.squeeze(0).cpu())
        plt.imshow(original_image)
        plt.axis('off')
        plt.show()

        # Constellation 모델을 사용해 이미지 인코딩
        r, mu_q, logvar_q, a, masks, learned_mask, recon, residue = model.encode(image)
        #print("r: ", r)
        #print("mu_q: ", mu_q)
        #print("logvar_q: ", logvar_q)
        #print("a: ", a)
        #print("o: ", masks)
        #print("learned_mask: ", learned_mask)
        #print("recon: ", recon)
        #print("residue: ", residue)

        # 사용자로부터 텍스트 프롬프트를 받아 처리
        new_label = input("텍스트 프롬프트를 입력해 모델을 조작하시오: ")
        new_labels = [label.strip() for label in new_label.strip('[]').split(',')] #자연어 멀티핫 라벨 분할
        one_hot_labels = process_labels(new_labels, num_classes)

        # 데이터에 붙어있는 기존 라벨 원핫인코딩으로 전환 #...........원본이미지를 원핫인코딩
        #one_hot_labels1 = process_labels1(label, word_to_idx1, num_classes1)

        # SCAN 모델을 이용하여 텍스트 임베딩 생성
        scan_embedding, _, mu0, logvar0 = scan(one_hot_labels)
        label_embedding, _, mu1, logvar1 = scan(one_hot_labels1)
        combinated_embedding,_,_ = recomb(mu1,logvar1,mu0,logvar0)

        print("scan_embedding:", scan_embedding)
        print("combinated_embedding:", combinated_embedding)

        # BVAE 디코더를 통해 새로운 잠재 벡터 생성
        new_r = bvae.decoder(combinated_embedding)

        #print("new_r:", new_r)

        # Constellation 모델 디코딩
        reconstructed_a, _, _ = model.decode(new_r)

        #print(reconstructed_a)

        # 복원된 마스크를 원래 이미지에 추가
        reconstructed_o = reconstructed_a + residue

        #print("reconstructed_o:", reconstructed_o)

        # MONet 모델을 사용해 최종 이미지 복원
        monet_output = monet(image) #.......... 원래 monet(image, masks)인데 gt가 아니라 monet모델을 이용한 output을 써보자,,,
        final_output = monet_output['reconstructions']

        # 최종 출력을 이미지로 변환
        final_image = transforms.ToPILImage()(final_output.squeeze(0).cpu().clamp(0, 1)) #ToPILImage: 텐서를 numpy/PILImage 형태로 바꾸기

        # TensorBoard에 이미지 기록
        # writer.add_image('Final Output', transforms.ToTensor()(final_image), idx) #ToTensor: numpy/PILImage 형태를 텐서형태로 바꾸기
    
        # 최종 이미지 출력
        print("조작 결과:")
        plt.imshow(final_image)
        plt.axis('off')
        plt.show()

        # break  
        # test num개의 이미지에 대해서 평가