
# Prerequisites

1. Python 3.11
2. pip install -r requirements.txt

# Code organization

The main entry point is level_set_traversal.py . There are two compulsory args: model and dataset. It can be run in two modes '--examples' or '--full_eval'.

1. '--examples': 
    This picks one image from each class in target_classes, and tries to find paths from each image to every other image using the LST algorithm
    It outputs a pkl file summarizing the results in 'plots_and_figures' directory, which can be loaded once generated with the '--load_dict' flag
    It also saves a visualization of the final LST output images and the triangle contour plot. 
    Enabling the 'get_widths' option also gets a visualization of the path widths.

2. '--full_eval': 
    This runs the LST algo over num_samples images picked from the dataset.
    Dumps a pkl file containing a summary of the LST results with distances (Tab 1 in paper), along with a pkl file containing stats over the triangular regions (Tab 2). The tables can be printed using print_tables.py
    

Parameters used in the main paper: python -u level_set_traversal.py resnet50-normal imagenet --full_eval --step_size 1e-2 --iters 400 --width_distances 0.5 1 1.5 2 --log_step 20 --pthresh 0.2 --adv_pert --adv_step_size 2e-3 --batch_size 196 --get_final_imgs"

TODOS for Vinu:
    Get all the above (viz image grid, contour plot, width plot, tab 1 and tab 2) for:
        (a) diff hyperparam values
        (b) diff target classes
        (c) CIFAR-10 

<!-- # Examples:  -->

<!-- ### ResNet-50 (normal) Image
hostname; python -u null_space_attack_imagenet_new.py resnet50-normal --full_eval --step_size 1e-2 --iters 400 --width_distances 0.5 1 1.5 2 --log_step 20 --pthresh 0.2 --adv_pert --adv_step_size 3e-4 --batch_size 196"

### ResNet-50 (Linf robust)
python -u null_space_attack_i
magenet_new.py resnet50-linf --full_eval --step_size 1e-2 --iters 400 --width_distances 0.5 1 1.5 2 --log_step 20 --pthresh 0.2 --adv_pert --adv_step_size 2e-3 --batch_size 196" -->
