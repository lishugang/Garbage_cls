######################################################
#----------------------coco2017----------------------#
######################################################
# deformable_detr
# map = 
CUDA_VISIBLE_DEVICES=4,5,6,7, GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh --epochs 40 --batch_size 2


# occ
# CUDA_VISIBLE_DEVICES=1,3,5,7 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh --epochs 999999 --batch_size 2





