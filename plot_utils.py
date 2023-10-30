import matplotlib.pyplot as plt
import torch
import pickle
from torchvision import transforms
import timm
from matplotlib.patches import Rectangle

import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_contour(x_dim, y_dim, x_steps, y_steps, scalar_field, file_path, v_min, v_max, levels=None):
    x, y = np.mgrid[-x_dim/2:x_dim/2:x_steps*1j, -y_dim/2:y_dim/2:y_steps*1j]
    cs = plt.tricontourf(x, y, scalar_field, zorder=1, cmap=cm.jet, extent=[-x_dim/2.0, x_dim/2.0, -y_dim/2.0, y_dim/2.0], vmin=v_min, vmax=v_max, levels=levels)
    plt.colorbar(cs)
    return cs.levels

def visualize_path_radius(x, y, radii, save_path):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    Z_2 = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = radii[j][i][0]
            Z_2[i][j] = radii[j][i][1]
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.plot_surface(X, -Y, Z, cmap='viridis')
    # ax.plot_surface(X, Y, Z_2, 50, cmap='inferno')
    # ax.plot_surface(X, -Y, Z_2, 50, cmap='inferno')

    ax.set_xlabel('x')
    ax.set_ylabel('L2 distance')
    ax.set_zlabel('Confidence')
    ax.view_init(60, -60)
    # if i == 2:
    #     ax.view_init(0,-90)
    plt.savefig(save_path)
    return

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
    edge_2_mask = torch.tensor(edge_2_mask).to(device)
    edge_1_mask = torch.tensor(edge_1_mask).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(normalizer(all_probe_imgs)), dim=1).cpu().numpy()
        probs = probs[:,source_class]
    
    return probs.mean(), probs[edge_1_mask].mean(), probs[edge_2_mask].mean()

def visualize_nplane(model, normalizer, imgs_list, source_class, save_path='./plots_and_figures/plane.png', num_interps=10, im_width=0.2):
    fig, ax = plt.subplots(ncols=len(imgs_list), figsize=[10*len(imgs_list), 10])
    levels = 100
    for ti, imgs in enumerate(imgs_list):
        source_coords = np.array((0,0))
        corner_coords_1 = source_coords + np.array((1,0))
        cos_sim = torch.nn.functional.cosine_similarity((imgs[2]-imgs[0]).reshape(-1), 
                                                        (imgs[1] - imgs[0]).reshape(-1), dim=0).item()
        mag = torch.norm(imgs[2]-imgs[0]).item()/(torch.norm(imgs[1]-imgs[0]).item())
        print(mag, cos_sim)
        corner_coords_2 = source_coords + np.array(( mag*cos_sim, mag*np.sqrt(1 - cos_sim**2) ))

        t1l = np.arange(num_interps+10)/num_interps
        t2l = np.arange(num_interps+10)/num_interps

        all_coords = []
        all_probe_imgs = []
        for t1 in t1l:
            for t2 in t2l:
                if 1 - t1 - t2 < 0:
                    continue
                all_coords.append((1-t1-t2)*source_coords + t1*corner_coords_1 + t2*corner_coords_2)
                all_probe_imgs.append((1-t1-t2)*imgs[0] + t1*(imgs[1]) + t2*(imgs[2]))

        all_coords = np.stack(all_coords)
        all_probe_imgs = torch.stack(all_probe_imgs).to(device)

        print(all_probe_imgs.shape)
        with torch.no_grad():
            probs = torch.softmax(model(normalizer(torch.clamp(all_probe_imgs,0,1))), dim=1).cpu().numpy()
            probs = probs[:,source_class]

        plt.tight_layout()
        ax[ti].set_axis_off()
        ax[ti].set_xlim(min(corner_coords_2[0], -im_width), max(corner_coords_2[0], 1+im_width))
        ax[ti].set_ylim(min(corner_coords_2[1], -im_width), corner_coords_2[1]+im_width)
        cf = ax[ti].tricontourf(all_coords[:,0], all_coords[:,1], probs, vmin=0, vmax=1, levels=levels)
        # if ti == 0:
        #     levels = cf.levels 
        ax[ti].plot(all_coords[:,0], all_coords[:,1], 'ro')
        if ti == 2:
            trucf = cf
        axins = ax[ti].inset_axes([source_coords[0]-im_width, source_coords[1]-im_width, im_width, im_width], transform=ax[ti].transData)
        axins.imshow(imgs[0].numpy().transpose((1, 2, 0)))
        axins.set_axis_off()
        axins = ax[ti].inset_axes([corner_coords_1[0], corner_coords_1[1]-im_width, im_width, im_width], transform=ax[ti].transData)
        axins.imshow(imgs[1].numpy().transpose((1, 2, 0)))
        axins.set_axis_off()
        axins = ax[ti].inset_axes([corner_coords_2[0]-im_width/2, corner_coords_2[1], im_width, im_width], transform=ax[ti].transData)
        axins.imshow(imgs[2].numpy().transpose((1, 2, 0)))
        axins.set_axis_off()
        
    # plt.colorbar(cf, ax=ax)
    colorbar = plt.colorbar(trucf, ax=ax, ticks=[0, 0.5, 1], orientation='vertical')
    trucf.set_clim(0, 1)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return

def plot_images(model, normalizer, inp_per_class, classes, class_names, save_path, target_layer=None, show_grad=False, gradcam_fn=AblationCAM):
    softmax = torch.nn.Softmax(dim=-1)
    if show_grad:
        cam = gradcam_fn(model=model, target_layers=[target_layer], use_cuda=True)
    f, axarr = plt.subplots(len(classes),len(classes), figsize=(2*len(classes),2*len(classes))) 
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.4)
    for i in range(len(classes)):
        for j in range(len(classes)):
            
            if j == 0:
                axarr[i, j].set_ylabel(class_names[classes[i]])
            if i == 0:
                axarr[i, j].set_title(class_names[classes[j]])

            if show_grad:
                grayscale_cam = cam(input_tensor=normalizer(inp_per_class[i, j].unsqueeze(0).requires_grad_()), target_category=j)
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(inp_per_class[i, j].clamp(0,1).numpy().transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
                axarr[i, j].imshow(visualization)
            else:
                 axarr[i, j].imshow(inp_per_class[i, j].numpy().transpose((1, 2, 0))) 
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_yticks([])
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_xticks([])
            #axarr[i, j].set_xlabel(np.round(dist_per_class[i, j].squeeze(), 3))
            conf = float(softmax(model(normalizer(inp_per_class[i, j].unsqueeze(0).to(device))))[0][classes[j]])
            axarr[i, j].set_xlabel(np.round(conf, 4))
            if i == j:
                axarr[i, j].patch.set_edgecolor('black')  
                axarr[i, j].patch.set_linewidth(5)  
            if i==0 and j == 0 and len(classes) != 10:
                bbox = axarr[i, j].get_position()
                rect = Rectangle((.2*bbox.width+bbox.height,0.7*bbox.width), 1.1*bbox.width, 0.8 + 0.2*bbox.width, edgecolor='blue', facecolor='white', zorder=-1, transform=f.transFigure, clip_on=False, linewidth=4)
                axarr[i, j].add_artist(rect)
    for ax in axarr.flat:
        ax.patch.set_visible(False)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # if show_grad:
    #     plt.savefig(f'./{suffix}_imagenet_examples_gradcam.pdf')
    # else:
    #     plt.savefig(f'./{suffix}_examples.pdf')  
