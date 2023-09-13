cd /home/jh/disk/workspace/DeepfakeBench/training
GPUS=(2)
CONFIG_FILE=./config/detector/rgbmsnlc.yaml
# rgbmsnlc
TASK_NAME=rgbmsnlc_DA
COMPRESSION=c40
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-NT --gpus ${GPUS[*]} --task_name $TASK_NAME\_NT_c23
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-F2F --gpus ${GPUS[*]} --task_name $TASK_NAME\_F2F_c23
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-DF --gpus ${GPUS[*]} --task_name $TASK_NAME\_DF_c23
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-FS --gpus ${GPUS[*]} --task_name $TASK_NAME\_FS_c23
python train.py --detector_path $CONFIG_FILE  --train_dataset FaceForensics++ --gpus ${GPUS[*]} --task_name $TASK_NAME\_FF_all_$COMPRESSION --compression $COMPRESSION

