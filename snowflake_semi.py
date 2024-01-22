import sys
sys.path.insert(0,"/global/homes/z/zhangtao/tsi-cloud/Semi-supervised-learning")
import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn import BasicDataset
import sys
from torchvision import datasets,transforms
import torch
from semilearn.datasets.augmentation import RandAugment
import matplotlib.pyplot as plt
import numpy as np
from semilearn.datasets.utils import split_ssl_data
import argparse

def transform_fun():
    n = 224
    #n = 32
    crop_ratio = config.crop_ratio

    transform_eval = transforms.Compose([transforms.Resize((n,n)),transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize((n,n))])
    transform_weak = transforms.Compose([
        transforms.Resize(n),
        transforms.RandomCrop(n, padding=int(n * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(n),
        transforms.RandomCrop(n, padding=int(n * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor()
    ])

    return transform_eval, transform, transform_weak, transform_strong

def train_main(train_path):
    # ## Step 3: create dataset

    # cifar.py get_cifar()
    #lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data()
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)
    train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
    train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
    eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size, data_sampler=None)

    # ### 3.2 train dataloader

    train_img = []
    train_target = []

    train_data = datasets.ImageFolder(f'{train_path}',transform)

    for img, target in train_data:
        train_img.append(img)
        train_target.append(target)

    train_img = np.array(train_img)
    train_target = train_target

    lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(config, train_img, train_target,
                                                          num_classes, config.num_labels,
                                                          include_lb_to_ulb=config.include_lb_to_ulb)

    del train_img, train_target

    lb_dset = BasicDataset(config.algorithm, lb_data, lb_target, num_classes, transform_weak, False, transform_strong, False)
    ulb_dset = BasicDataset(config.algorithm, ulb_data, ulb_target, num_classes, transform_weak, True, transform_strong, False)

    train_lb_loader = get_data_loader(config, lb_dset, config.batch_size)
    train_ulb_loader = get_data_loader(config, ulb_dset, int(config.batch_size * config.uratio))

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_target:
        lb_count[c] += 1
    for c in ulb_target:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    del lb_data, lb_target, ulb_data, ulb_target
    del lb_dset, ulb_dset

    # ### 3.3 test dataloader

    test_img = []
    test_target = []

    test_data = datasets.ImageFolder(f'{test_path}',transform)

    for img, target in test_data:
        test_img.append(img)
        test_target.append(target)

    test_img = np.array(test_img)
    test_target = test_target

    eval_dset = BasicDataset(config.algorithm, test_img, test_target, num_classes, transform_eval, False, None, False)
    eval_loader = get_data_loader(config, eval_dset, config.eval_batch_size, data_sampler=None)
    del test_img, test_target, eval_dset

    # ## Step 4: train

    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader,eval_loader)


def test_main(test_path):
    # load trained model
    trainer = Trainer(config, algorithm)
    trainer.algorithm.load_model('../saved_models/fixmatch/model_best.pth')
    #trainer.algorithm.load_model('../saved_models/fixmatch/latest_model.pth')

    # ### 3.3 test dataloader

    test_img = []
    test_target = []

    test_data = datasets.ImageFolder(f'{test_path}',transform)

    for img, target in test_data:
        test_img.append(img)
        test_target.append(target)

    test_img = np.array(test_img)
    test_target = test_target


    eval_dset = BasicDataset(config.algorithm, test_img, test_target, num_classes, transform_eval, False, None, False)
    eval_loader = get_data_loader(config, eval_dset, config.eval_batch_size, data_sampler=None)

    trainer.evaluate(eval_loader)
    y_pred, y_logits, y_true = trainer.predict(eval_loader, return_gt=True)
    np.savez('fixmatch.npz',y_pred=y_pred, y_true=y_true)

# ## Step 1: define configs and create config


config = {
    'algorithm': 'fixmatch',
    'net': 'vit_small_patch16_224',
    'use_pretrain': True, 
    'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_small_patch16_224_mlp_im_1k_224.pth',
    'save_dir': '../saved_models/',

    # optimization configs
    'epoch': 200,  
    'num_train_iter': 2000,  
    'num_eval_iter': 500,  
    'num_log_iter': 50,  
    'optim': 'AdamW',
    'lr': 2e-4,
    'layer_decay': 0.5,
    'batch_size': 64,
    'eval_batch_size': 64,


    # dataset configs
    'dataset': 'cifar10',
    'num_labels': 1000,
    'num_classes': 5,
    'img_size': 224,
    'crop_ratio': 0.875,
    'data_dir': '../data',
    'ulb_samples_per_class': None,

    # algorithm specific configs
    'hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 2,
}
config = get_config(config)
parser = argparse.ArgumentParser()
parser.add_argument('-p','--process',required=True)

args = parser.parse_args()
process = args.process

torch.manual_seed(42)
np.random.seed(42)

target_class = ['AG','CC','GR','PC','SP']
num_classes = len(target_class)
train_path = f'/pscratch/sd/z/zhangtao/storm/mpc/key_paper/training'
test_path  = f'/pscratch/sd/z/zhangtao/storm/mpc/key_paper/test'

# ## Step 2: create model and specify algorithm
algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

transform_eval, transform, transform_weak, transform_strong = transform_fun()

if process == 'train':
	train_main(train_path)

if process == 'test':
	test_main(test_path)




