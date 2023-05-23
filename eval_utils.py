import matplotlib.pyplot as plt
import torch
import pickle
from torchvision import transforms
import timm
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_avg_conf(model, normalizer, imgs, source_class, num_interps=10):
    t1l = np.arange(num_interps+1)/num_interps
    t2l = np.arange(num_interps+1)/num_interps

    all_probe_imgs = []
    edge_1_mask = []
    edge_2_mask = []
    for t1 in t1l:
        for t2 in t2l:
            if 1 - t1 - t2 < 0:
                continue
            edge_1_mask.append(t2==0)
            edge_2_mask.append(t1==0)
            all_probe_imgs.append((1-t1-t2)*imgs[0] + t1*(imgs[1]) + t2*(imgs[2]))

    all_probe_imgs = torch.stack(all_probe_imgs).to(device)
    edge_2_mask = np.array(edge_2_mask)
    edge_1_mask = np.array(edge_1_mask)

    with torch.no_grad():
        probs = torch.softmax(model(normalizer(all_probe_imgs)), dim=1).cpu().numpy()
        probs = probs[:,source_class]
    p_src = probs[edge_2_mask & edge_1_mask]
    assert len(p_src) == 1
    p_src = p_src[0]
    return p_src, probs.mean(), [(probs > p_src - d).mean() for d in [0.0, 0.1, 0.2, 0.3]], np.minimum((probs[edge_1_mask]-p_src).min(), (probs[edge_2_mask]-p_src).min()), np.concatenate((probs[edge_1_mask], probs[edge_2_mask]), axis=0).mean()

def get_avg_conf_for_img_set(model, normalizer, img_dicts, num_interps=10, trg_classes=[(0,1), (2,3), (4,0)]):
    all_metrics = []
    for i, (src_img, src_label) in enumerate(zip(img_dicts[0]['source_images'], img_dicts[0]['labels'])):
        metrics = []
        for trgclass1, trgclass2 in trg_classes:
            srcp, triangle_mean, triangle_frac, edge_min, edge_mean = get_avg_conf(model, normalizer, [src_img, 
                                                                                                    img_dicts[trgclass1]['final_images'][i], 
                                                                                                    img_dicts[trgclass2]['final_images'][i]], 
                                                                            src_label, num_interps=num_interps)
            metrics.append((srcp, triangle_mean, *triangle_frac, edge_mean, edge_min ))
        metrics = np.array(metrics)
        mean_metrics = metrics[:,:-1].mean(axis=0)
        min_metrics = metrics[:,-1:].min(axis=0)
        all_metrics.append( np.concatenate((mean_metrics, min_metrics)))
    all_metrics = np.stack(all_metrics)
    print(all_metrics.mean(axis=0), all_metrics.std(axis=0))
    return all_metrics
            
