import cv2
from torchvision import datasets,transforms
import sys
import numpy as np
from skimage import metrics
import multiprocessing
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image

def imageEncoder(img):
    #img1 = Image.fromarray(img).convert('RGB')
    img1 = Image.fromarray(img)
    img1 = preprocess(img1).unsqueeze(0)
    img1 = model.encode_image(img1)
    return img1
def generateScore(image1, image2):
    #test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    #data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score


train_path = f'/pscratch/sd/z/zhangtao/storm/tmp/key_paper/training'
test_path  = f'/pscratch/sd/z/zhangtao/storm/tmp/key_paper/test'

n=224
transform = transforms.Compose([transforms.Resize((n,n))])

train_data = datasets.ImageFolder(f'{train_path}',transform)
test_data = datasets.ImageFolder(f'{test_path}',transform)

# load data
train_img_1 = []
train_target_1 = []
train_img_2 = []
train_target_2 = []

for img, target in train_data:
    if target == 0:
        train_img_1.append(img)
        train_target_1.append(target)
    else:
        train_img_2.append(img)
        train_target_2.append(target)


test_img_1 = []
test_target_1 = []
test_img_2 = []
test_target_2 = []

for img, target in test_data:
    if target == 0:
        test_img_1.append(img)
        test_target_1.append(target)
    else:
        test_img_2.append(img)
        test_target_2.append(target) 

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
testing_data = test_img_1

def similarity(data):
    start_ix,n_len = int(data[0]),int(data[1])
    print(f'{start_ix=},{n_len=}')

    total_score = 0

    for j in range(start_ix,start_ix+n_len):

        img1 = np.array(testing_data[j])

        ssim_score_1 = 0
        ssim_score_2 = 0

        print(f'{start_ix=},{n_len=}, {j=}')

        for i in range(3):
            img2 = np.array(train_img_1[j])
            score = generateScore(img1, img2)
            ssim_score_1 += score

            img2 = np.array(train_img_2[i])
            score = generateScore(img1,img2)
            ssim_score_2 += score

        if ssim_score_1 > ssim_score_2:
            total_score += 1

    print(f'{total_score=}')

nprocess=10
len_seq = len(testing_data)
num = len_seq//nprocess
start_ix = np.arange(len_seq)[::num]
nn = np.zeros(nprocess)
for i in range(len(nn)):
    if i != len(nn) - 1:
        nn[i] = start_ix[i+1] - start_ix[i]
    else:
        nn[i] = len_seq - start_ix[i]

input_data = list(zip(start_ix,nn))
pool = multiprocessing.Pool(nprocess)
pool.map(similarity,input_data)
