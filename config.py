# License: MIT
# Author: Karl Stelzner

from collections import namedtuple
import os

config_options = [
    # Training config
    'vis_every',  # Visualize progress every X iterations
    'batch_size',
    'num_epochs',
    'load_parameters',  # Load parameters from checkpoint
    'checkpoint_file',  # File for loading/storing checkpoints
    'data_dir',  # Directory for the training data
    'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
    'beta', #내가 추가
    'gamma' #내가 추가
]

MonetConfig = namedtuple('MonetConfig', config_options)

sprite_config = MonetConfig(vis_every=50,
                            batch_size=64, #논문: 64
                            num_epochs=20, #논문에서는 500000 iteration 진행 -> batch_size 64면 epoch=64
                            load_parameters=True,
                            checkpoint_file='./checkpoints/sprites.ckpt',
                            data_dir='./data/',
                            parallel=True,
                            num_slots=9, #constellation dataset에 맞춰서 늘림
                            num_blocks=5,
                            channel_base=128, #64
                            bg_sigma=0.09, #background sigma, 1번째 슬롯에만 쓰이는 파라미터, 잘 조절하면 slot1에 배경색만 나오게된다. (논문: 0.09)
                            fg_sigma=0.11, #논문: 0.11
                            beta=8.0, #논문 0.5
                            gamma=0.25 #논문 0.25
                           )


