## ResNet-50 (normal)
srun --qos=default --partition=tron --account=nexus --time=4:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=32gb bash -c "hostname; python -u null_space_attack_imagenet_new.py resnet50-normal --full_eval --step_size 1e-2 --iters 400 --width_distances 0.5 1 1.5 2 --log_step 20 --pthresh 0.2 --adv_pert --adv_step_size 3e-4 --batch_size 196"

## ResNet-50 (Linf robust)
srun --qos=default --partition=tron --account=nexus --time=4:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=32gb bash -c "hostname; python -u null_space_attack_i
magenet_new.py resnet50-linf --full_eval --step_size 1e-2 --iters 400 --width_distances 0.5 1 1.5 2 --log_step 20 --pthresh 0.2 --adv_pert --adv_step_size 2e-3 --batch_size 196"
