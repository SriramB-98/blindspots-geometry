
import pickle
import torch
import timm
import numpy as np
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'tv_resnet50'
model = timm.create_model(model_name, pretrained=True)
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for p in model.parameters():
    p.requires_grad_(False)
model.eval()
model = torch.nn.Sequential(normalizer, model).to(device)
batch_size = 196

prefix = './plots_and_figures/resnet50-normal_imagenet_640-133-485-937-618'
hyperparam_str = '1000samples_stepsize-0.01_400iters_advstepsize-0.002'

with open(f'{prefix}_{hyperparam_str}_fulleval_dict.pkl', 'rb') as fp:
    _, dict_per_target = pickle.load(fp)

all_probs = []
for dict in dict_per_target:
    imgs = dict['final_images']
    labels = dict['labels']
    all_probs = []
    for i in tqdm(range(len(imgs)//batch_size + 1)):
        img_batch = imgs[i*batch_size:(i+1)*batch_size]
        label_batch = labels[i*batch_size:(i+1)*batch_size]
        img_batch = img_batch.to(device)
        logits = model(img_batch)
        probs = torch.nn.functional.softmax(logits, dim=1)[torch.arange(len(img_batch)), label_batch]
        all_probs.append(probs.detach().cpu())
all_probs = torch.cat(all_probs, dim=0)
print(all_probs.mean().item(), all_probs.std().item())
        