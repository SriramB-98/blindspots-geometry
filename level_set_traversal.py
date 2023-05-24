#!/usr/bin/env python
import pickle
import torch
torch.manual_seed(0)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
random.seed(0)
#random.seed(1)
import argparse
import timm
import robustness
import torchvision
from plot_utils import *
from eval_utils import *
from robustbench import load_model
import torchvision.transforms as transforms
from torchmetrics.functional import structural_similarity_index_measure
import lpips

plt.tight_layout()

import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',  type=str, help='Model name')
    parser.add_argument('dataset',  type=str, help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='number of workers')
    parser.add_argument('--step_size', type=float, default=5e-2,
                        help='batch size')
    parser.add_argument('--pthresh', type=float, default=0.05)
    parser.add_argument('--imgnet_path', type=str, default="~/ILSVRC2012",)
    parser.add_argument('--cifar_path', type=str, default="~/Cifar10",)
    parser.add_argument('--iters', type=int, default=100,
                        help='batch size')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Num samples for full eval')
    parser.add_argument('--num_images', type=int, default=100,
                        help='number of images per class')
    parser.add_argument('--log_step', type=int, default=10,)
    parser.add_argument('--examples', action='store_true',
                        help='example images')
    parser.add_argument('--get_widths', action='store_true',
                        help='Get widths')
    parser.add_argument('--get_final_imgs', action='store_true',
                        help='Get final images')
    parser.add_argument('--full_eval', action='store_true',
                        help='example images')
    parser.add_argument('--show_grad', action='store_true',
                        help='Grad Cam')
    parser.add_argument('--use_logit_init', action='store_true',
                        help='Grad Cam')
    parser.add_argument('--get_confs_over_path', action='store_true',
                        help='Get confs over path')
    parser.add_argument('--get_images', action='store_true',
                        help='Get images over path')
    parser.add_argument('--width_distances', nargs='+', type=float, default=[5, 10, 15, 20, 25, 30],
                        help='width distances')
    parser.add_argument('--target_classes', nargs='+', type=int, default=[99, 199, 299, 400, 499],
                        help='target classes')
    parser.add_argument('--load_dict', action='store_true')
    parser.add_argument('--adv_pert', action='store_true',
                    help='whether to use adversarial perturbation')
    parser.add_argument('--adv_step_size', type=float, default=1/255.0,
                    help='step size of adversarial perturbation')
    parser.add_argument('--load_model_path', type=str, default=None,
                    help='path to load model')
    parser.add_argument('--desc', type=str, default="",
                        help='number of images per class')
    args = parser.parse_args()
    return args

def prod(x):
    p = 1
    for i in x:
        p = p*i
    return p

def dict_combine(d1, d2):
    for k in d2:
        if d2[k] is None:
            continue
        if k in d1:
            if d1[k] is None:
                continue
            d1[k] = torch.cat((d1[k], d2[k]), dim=0)
        else:
            d1[k] = d2[k]
    return d1

def measure_width(img, model, normalizer, index, dir_vec, distances = [5, 10, 15, 20, 25, 30], num_vecs=256):
    # print('Entered measure_width')
    softmax = torch.nn.Softmax(dim=-1)
    init_confidence = softmax(model(normalizer(img.clamp(0,1))))[:, index]
    rand_vecs = torch.randn(num_vecs, *img.shape[1:]).to(init_confidence.device)
    if torch.linalg.vector_norm(dir_vec) == 0:
        return np.zeros((len(distances)+1, 2))
    dir_vec = dir_vec/torch.linalg.vector_norm(dir_vec)
    a = torch.sum(rand_vecs*dir_vec, dim=(1,2,3))
    rand_vecs = rand_vecs - torch.einsum("ij, jklm -> iklm" ,a.unsqueeze(-1), dir_vec)
    rand_vecs = rand_vecs/torch.linalg.vector_norm(rand_vecs, dim=(1,2,3), keepdim=True)
    confs = [torch.Tensor((float(init_confidence), float(init_confidence)))]
    for i in distances:
        # print(i)
        pert_imgs = img + i*rand_vecs
        with torch.no_grad():
            pert_confs = softmax(model(normalizer(pert_imgs.clamp(0,1))))[:, index]
#         min_confs, max_confs = float(torch.min(pert_confs)), float(torch.max(pert_confs))
        confs.append(torch.quantile(pert_confs, q=torch.Tensor([0.05, 0.95]).to(pert_confs.device) ).cpu())
    confs = torch.stack(confs)
    return confs

def get_targets(model_normalizers, loader, target_classes, device):
    target_images = dict()
    orig_target_classes = target_classes.copy()
    for i, data_batch in enumerate(loader):
        imgs, labels = data_batch
        rel_inds = (labels[:, None] == torch.LongTensor(target_classes)[None,:]).any(dim=-1)
        if not rel_inds.any():
            continue
        imgs = imgs[rel_inds].to(device)
        labels = labels[rel_inds].to(device)
        hits = torch.ones(len(imgs), dtype=torch.bool).to(device)
        for model, normalizer in model_normalizers:
            with torch.no_grad():
                pred = model(normalizer(imgs))
            probs = torch.softmax(pred, dim=-1)
            pred_labels = torch.argmax(pred, dim=-1)
            hits = hits & (pred_labels == labels) & (probs[torch.arange(len(pred_labels)), labels] > 0.6) # img is correctly classified and confidence is greater than 0.6
        rel_imgs = imgs[hits]
        rel_labels = labels[hits]
        for j, (img, label) in enumerate(zip(rel_imgs, rel_labels)):
            label = label.cpu().item()
            if label in target_classes:
                target_images[label] = img.cpu()
                target_classes.remove(label)
                print(target_images.keys())
                if len(target_classes) == 0:
                    return [target_images[k] for k in orig_target_classes]
    return [target_images[k] for k in orig_target_classes]

def get_nullspace_projection(J, v):
    y_hat = torch.sum(J*v, -1, keepdim=True)/torch.sum(J * J, -1,  keepdim=True)
    x = v - (J * y_hat)
    # computing othogonal projection for pert with positive inner product as well
    return x

def l2(x, y):
    return torch.sum((x - y)**2, dim=(1,2,3)).sqrt()

def linf(x, y):
    return torch.max(torch.abs(x - y).reshape(len(x),-1), dim=-1).values

def ssim(x, y):
    y = y.expand(len(x),-1,-1,-1)
    return structural_similarity_index_measure(x, y, data_range=1.0, reduction='none')

lpips_alex = lpips.LPIPS(net='alex').to(device)
def lpips_dist(x, y):
    y = y.expand(len(x),-1,-1,-1)
    return lpips_alex(2*x-1, 2*y-1).squeeze()

def batch_level_set_traversal(model, normalizer, dataset, source_classes, target_image, target_class, stepsize, iterations, device, 
                           pthresh=0.05, inp=None, get_widths=False, get_confs_over_path=False, get_images=False, get_final_imgs=False, log_step=10, dfunc_list=[]):

    source_classes = source_classes.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    if inp is None:
        inp = torch.Tensor(dataset).to(device)#.requires_grad_(True)
    else:
        inp = inp.detach().clone().to(device)#.requires_grad_(True)
    mask = torch.ones(*inp.shape, dtype=torch.bool).to(device)
    target_image = target_image.to(device)
    indices = torch.arange(len(inp))
    with torch.no_grad():
        pred = model(normalizer(inp.clamp(0,1)))
    init_probs = softmax(pred)
    print(source_classes, init_probs[indices, source_classes])
    init_labels = torch.argmax(init_probs, dim=-1)
    rel_img_mask = (init_labels == source_classes).cpu()
    # all_dataset = dataset
    # all_source_classes = source_classes
    dataset = dataset[rel_img_mask]
    source_classes = source_classes[rel_img_mask]
    inp = inp[rel_img_mask]
    indices = torch.arange(len(inp))

    widths_over_path, inp_over_path, confs_over_path, confs_over_path_target = [], [], [], []
    adv_delta = None
    init_target_layer_acts = source_classes
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    inp = inp + (4/255)*(torch.rand_like(inp)-0.5)
    inp = torch.clamp(inp, 0., 1.)
    inp.requires_grad_()


    for i in range(iterations):

        pred = model(normalizer(inp))
        probs = softmax(pred)
        target_activations = pred
        cost = loss(target_activations, init_target_layer_acts)
        cost.backward()
        J = inp.grad.reshape(len(probs), prod(inp.shape[1:]))
        v = target_image.flatten() - inp.flatten(start_dim=1)
        null_vec = get_nullspace_projection(J, v).reshape(inp.shape)

        if adv_delta is None:
            adv_delta = (args.adv_step_size/2) * J
        else:
            adv_delta = adv_delta + (args.adv_step_size/2) * J
        adv_delta = torch.clamp(adv_delta, -args.adv_step_size, args.adv_step_size)
                
        if i%log_step == 0:
            if get_widths:
                print(f'Getting widths at {i}')
                with torch.no_grad():
                    widths_at_i = []
                    for j in range(len(inp)):
                        if probs[j, source_classes[j]] < pthresh:
                            widths_at_i.append(np.zeros((len(args.width_distances)+1,2)))
                        else:
                            widths_at_i.append(measure_width(inp[j:j+1], model, normalizer, source_classes[j], null_vec[j:j+1], distances=args.width_distances))
                    widths_over_path.append(torch.stack(widths_at_i))
            if get_images:
                with torch.no_grad():
                    inp_over_path.append(inp.detach().cpu())

            if get_confs_over_path:
                confs_at_i = probs[indices, source_classes].detach().cpu()
                confs_at_i_target = probs[:, target_class].detach().cpu()
                confs_over_path.append(confs_at_i)
                confs_over_path_target.append(confs_at_i_target)

        if args.adv_pert:
            new_inp = inp + stepsize*null_vec - adv_delta.reshape(inp.shape)
        else:
            new_inp = inp + stepsize*null_vec
        new_inp = torch.clamp(new_inp, 0.0, 1.0)
        
        with torch.no_grad():
            pred_probs = softmax(model(normalizer(new_inp)))

        mask = ((init_probs[indices,source_classes] - pred_probs[indices,source_classes] < pthresh)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if ~ (mask.any()):
            print("Iterations stopped at:", i)
            break
        inp = (inp*(~mask) + new_inp*mask).detach()
        inp.requires_grad_()
    
    inp = inp.detach()
    with torch.no_grad():
        pred = model(normalizer(inp.clamp(0,1)))
    probs = softmax(pred)
    print(probs[indices, source_classes])
    with torch.no_grad():
        if len(dfunc_list) == 0:
            dist_target = None
        else:
            dist_target = torch.stack([dfunc(inp, target_image[None,:]).squeeze().detach().cpu() for dfunc in dfunc_list], dim=1)

    return {
        'source_images': dataset.cpu() if get_final_imgs else None,
        'final_images': inp.cpu() if get_final_imgs else None,
        'distances': dist_target,
        'widths_over_path': torch.stack(widths_over_path, dim=1) if get_widths else None,
        'confs_over_path': torch.stack(confs_over_path, dim=1) if get_confs_over_path else None,
        'confs_over_path_target': torch.stack(confs_over_path_target, dim=1) if get_confs_over_path else None,
        'imgs_over_path': torch.stack(inp_over_path, dim=1) if get_images else None,
        'labels': source_classes,
        }


def run_lst(model, normalizer, dataset , target_images, target_classes, device, pthresh=0.05, get_widths=False, get_final_imgs=False,
               dfunc_list=[], get_images=False, get_confs_over_path=False, num_samples=1000):
    all_return_dict = []
    for i, image in enumerate(target_images):
        print(i)
        accum_dict = dict([])
        for b, data_batch in enumerate(tqdm(dataset)):
            return_dict  = batch_level_set_traversal(model, normalizer, data_batch[0], data_batch[1],
                                                    image, target_classes[i], args.step_size, args.iters, device, 
                                                    pthresh=pthresh, get_widths=get_widths, get_images=get_images, get_final_imgs=get_final_imgs,
                                                    dfunc_list=dfunc_list, get_confs_over_path=get_confs_over_path, log_step=args.log_step)
            accum_dict = dict_combine(accum_dict, return_dict)
            if len(accum_dict['labels']) >= num_samples:
                break
        all_return_dict.append(accum_dict)
    return all_return_dict

def load_model_normalizer(model_name, model_type, dataset='imagenet'):
    if dataset == 'imagenet':
        if model_type == 'linf':
            if model_name == 'resnet50':
                normalizer, model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf').to(device)
            elif model_name == 'deit_small_patch16_224':
                normalizer, model = load_model(model_name='Singh2023Revisiting_ViT-S-ConvStem', dataset='imagenet', threat_model='Linf').to(device)
        else:
            model = timm.create_model(model_name, pretrained=True).to(device)
            normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        for p in model.parameters():
            p.requires_grad_(False)
    elif dataset == 'cifar10':
        if model_type == 'linf':
            model = load_model(model_name='Wu2020Adversarial_extra', dataset='cifar10', threat_model='Linf').to(device)
        else:
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(device)
        normalizer = lambda x: x
        for p in model.parameters():
            p.requires_grad_(False)
    model.eval()
    return model, normalizer

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.isdir('./plots_and_figures'):
        os.mkdir('./plots_and_figures')

    # dataset creation
    print("Loading data...")

    if args.dataset == 'imagenet':
        transform_fn = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                        ])
        dataset = torchvision.datasets.ImageNet(args.imgnet_path, split='val', 
                                                transform=transform_fn)
        
        dataset, _ = torch.utils.data.random_split(dataset, [len(dataset), 0])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                    shuffle=False, num_workers=args.num_workers)
        classes = torch.load("imagenet_classnames.pth")
    elif args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=args.cifar_path, train=False, download=True, transform=transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        classes = ['airplane',
                    'automobile',
                    'bird',
                    'cat',
                    'deer',
                    'dog',
                    'frog',
                    'horse',
                    'ship',
                    'truck']

    # load model
    print("Loading model...")
    
    model_name, model_type = args.model.split('-')  
    model, normalizer = load_model_normalizer(model_name, model_type, dataset=args.dataset)

    # get target imgs
    print("Getting target images...")
    if args.target_classes is None:
        args.target_classes = [random.randint(0,len(classes)) for _ in range(10)]
    target_classes = args.target_classes 
    target_classes_str = "-".join([str(x) for x in target_classes])

    try:
        target_images = []
        to_tensor = transforms.ToTensor()
        for i, tc in enumerate(target_classes):
            target_images.append(to_tensor(Image.open(f'./plots_and_figures/{args.dataset}_target_{tc}.png')))
            target_images[-1] = target_images[-1][:3]
        target_images = torch.stack(target_images, dim=0)
        with torch.no_grad():
                preds = torch.softmax(model(normalizer(target_images.to(device))), dim=-1).cpu().detach().numpy()
        print([preds[i, target_classes[i]] for i in range(len(target_classes))])
    except FileNotFoundError:
        if args.dataset == 'imagenet':
            all_models_normalizers = [load_model_normalizer(model_name, model_type, dataset='imagenet') for model_name in ['resnet50', 'deit_small_patch16_224'] for model_type in ['linf', 'normal']]
        elif args.dataset == 'cifar10':
            all_models_normalizers = [load_model_normalizer(model_name, model_type, dataset='cifar10') for model_name in ['wideresnet'] for model_type in ['linf', 'normal']]
        target_images = get_targets(all_models_normalizers, data_loader, target_classes.copy(), device)
        target_images = torch.stack(target_images, dim=0)
        for model, normalizer in all_models_normalizers:
            with torch.no_grad():
                preds = torch.softmax(model(normalizer(target_images.to(device))), dim=-1).cpu().detach().numpy()
            print([preds[i, target_classes[i]] for i in range(len(target_classes))])
        del all_models_normalizers
        # save target images
        print("Saving target images...")
        for i, img in enumerate(target_images):
            img = img.permute(1, 2, 0).numpy()
            print(img.shape)
            plt.imsave(f'./plots_and_figures/{args.dataset}_target_{target_classes[i]}.png', img)

    # attack chosen examples
    if args.examples:
        print("Running attack on examples...")
        example_loader = [(target_images,  torch.LongTensor(target_classes))]
        if not args.load_dict:
            dict_per_target = run_lst(model, normalizer, example_loader, target_images, target_classes, device,
                                      pthresh=args.pthresh, get_images=args.get_images, get_widths=args.get_widths,  
                                      get_final_imgs=True, dfunc_list=[], get_confs_over_path=args.get_confs_over_path)
            for i in range(len(dict_per_target)):
                dict_per_target[i]['final_images'][i] = target_images[i]
            with open(f'./plots_and_figures/{args.model}_{args.dataset}_{target_classes_str}_examples_dict.pkl', 'wb+') as fp:
                pickle.dump((vars(args), dict_per_target), fp)
        else:
            with open(f'./plots_and_figures/{args.model}_{args.dataset}_{target_classes_str}_examples_dict.pkl', 'rb') as fp:
                args, dict_per_target = pickle.load(fp)
                args = argparse.Namespace(**args)
                target_classes = args.target_classes

        inp_per_class = torch.stack([dict_per_target[i]['final_images'] for i in range(len(dict_per_target))])

        # Plot image grid
        plot_images(model, normalizer, inp_per_class, target_classes, classes, f"./plots_and_figures/{args.model}_{args.dataset}_{target_classes_str}_attacked_images.png")

        # Plot triangles for all target pairs
        all_trg_pairs = [(1,2), (2,3), (3,4), (1,3), (1,4), (2,4)]
        all_imgs = []
        for source_target_pairs in [[(0,0), (0,trg1), (0,trg2)] for trg1, trg2 in all_trg_pairs]: #must only have 3 elements, first must be original image
            imgs = []
            sc, tc1, tc2 = [target_classes[i] for _, i in source_target_pairs]
            for source_class, target_class in source_target_pairs:
                img = dict_per_target[target_class]['final_images'][source_class]
                imgs.append(img)
            all_imgs.append(imgs)
        visualize_nplane(model, normalizer, all_imgs, target_classes[source_class], save_path=f'./plots_and_figures/{args.model}_{args.dataset}_plane_src-{sc}.png', num_interps=10)

        if args.get_widths:
            for src_ind, dst_ind in [(0, 1), (2,3), (4,0)]:
                visualize_path_radius(np.arange(0, args.iters, args.log_step), np.array([0]+args.width_distances), dict_per_target[dst_ind]['widths_over_path'][src_ind], f"./plots_and_figures/{args.model}_{args.dataset}_src-{target_classes[src_ind]}_dst-{target_classes[dst_ind]}_path_widths.png")
       
    # attack all examples 
    if args.full_eval:
        if not args.load_dict:
            print(f"Running attack on {args.num_samples} correctly classified source examples...")
            dict_per_target = run_lst(model, normalizer, data_loader, target_images, target_classes, device, num_samples=args.num_samples,
                                     pthresh=args.pthresh, get_images=args.get_images, get_widths=args.get_widths, get_final_imgs=args.get_final_imgs, 
                                     dfunc_list=[l2, linf, ssim, lpips_dist], get_confs_over_path=args.get_confs_over_path)
            with open(f'./{args.model}_{args.num_samples}samples_stepsize-{args.step_size}_iters-{args.iters}_advstepsize{args.adv_step_size}_{args.dataset}_fulleval_dict.pkl', 'wb+') as fp:
                pickle.dump((vars(args), dict_per_target), fp)
        else:
            with open(f'./{args.model}_{args.num_samples}samples_stepsize-{args.step_size}_iters-{args.iters}_advstepsize{args.adv_step_size}_{args.dataset}_fulleval_dict.pkl', 'rb') as fp:
                args, dict_per_target = pickle.load(fp)
                args = argparse.Namespace(**args)
                target_classes = args.target_classes
        
        all_trg_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (1,3), (1,4), (2,4)]
        avg_confs = get_avg_conf_for_img_set(model, normalizer, dict_per_target, trg_classes=all_trg_pairs)
        with open(f'./{args.model}_{args.num_samples}samples_stepsize-{args.step_size}_iters-{args.iters}_advstepsize{args.adv_step_size}_{args.dataset}_all_confs.pkl', 'wb+') as fp:
            pickle.dump(avg_confs, fp)
