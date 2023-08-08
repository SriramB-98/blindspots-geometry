import torch
import numpy as np
import pickle

typ = 'confs' # 'dist' or 'confs' corresponds to the tables in the paper
dataset = 'imagenet' # which dataset
hyperparam_str = '1000samples_stepsize-0.01_400iters_advstepsize-0.002' #other hyperparams
# target_classes_str = '869-970-173-369-978'
target_classes_str = '640-133-485-937-618'
sep = '|'
math_mode = '$'

model_name_form = ['ResNet-50 (Normal)', 'ResNet-50 (AT)', 'DeiT-S (Normal)', 'DeiT-S (AT)']
for j, model_name in enumerate(['resnet50-normal']):#, 'resnet50-linf', 'deit_small_patch16_224-normal', 'deit_small_patch16_224-linf']):
    prefix = f'./plots_and_figures/{model_name}_{dataset}_{target_classes_str}'
    if typ == 'confs':
        with open(f'{prefix}_{hyperparam_str}_all_confs.pkl', 'rb') as fp:
            avg_confs = pickle.load(fp)
        means, stds = np.round(avg_confs.mean(axis=0), 3), np.round(avg_confs.std(axis=0), 3)
        means = means[:-1]
        stds = stds[:-1]
    if typ == 'dist':
        with open(f'{prefix}_{hyperparam_str}_fulleval_dict.pkl', 'rb') as fp:
            all_dicts = pickle.load(fp)
            if isinstance(all_dicts, tuple):
                all_dicts = all_dicts[1]


        all_dists = []
        for all_dict in all_dicts:
            all_dists.append(all_dict['distances'])
        all_dists = torch.cat(all_dists, dim=0)

        dist_means, dist_stds = all_dists.mean(dim=0), all_dists.std(dim=0)
        means, stds = [], []
        for i in range(len(dist_means)):
            means.append(np.round(dist_means[i].item(), 3))
            stds.append(np.round(dist_stds[i].item(),3))

    print(model_name_form[j] + f' {sep} ' + f' {sep} '.join([f'{mean} $\pm$ {std}' for mean, std in zip(means, stds)]) + '\\\\')
