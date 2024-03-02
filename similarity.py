import cv2
from torchvision import datasets,transforms
import sys
import numpy as np
from skimage import metrics

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

total_score = 0
print(len(test_img_1))
for j in range(len(test_img_2)):

    img1 = np.array(test_img_2[j])

    ssim_score_1 = 0
    ssim_score_2 = 0

    for i in range(20):
        img2 = np.array(train_img_1[j])
        score = generateScore(img1, img2)
        #img2 = cv2.cvtColor(np.array(train_img_1[i]), cv2.COLOR_BGR2GRAY)
        #ssim_score,_ = metrics.structural_similarity(img1, img2, full=True)
        ssim_score_1 += score

        img2 = np.array(train_img_2[i])
        score = generateScore(img1,img2)
        #ssim_score,_ = metrics.structural_similarity(img1, img2, full=True)
        ssim_score_2 += score

    if ssim_score_1 > ssim_score_2:
        total_score += 1
    print(ssim_score_1,ssim_score_2)

print(f'{total_score=}')
print(total_score/len(test_img_2))
