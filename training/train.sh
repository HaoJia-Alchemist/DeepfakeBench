GPUS=(2)
CONFIG_FILE=./config/detector/efficientnetb4.yaml
# rgbmsnlc
TASK_NAME=rgbmsnlc
COMPRESSION=raw
#python train.py --detector_path ./config/detector/rgbmsnlc.yaml --train_dataset FaceForensics++ --gpus ${GPUS[*]} --task_name rgbmsnlc_FF-all_c40
#TASK_NAME=efficientnetb4
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-NT --gpus ${GPUS[*]} --task_name $TASK_NAME\_NT_raw
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-F2F --gpus ${GPUS[*]} --task_name $TASK_NAME\_F2F_raw
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-DF --gpus ${GPUS[*]} --task_name $TASK_NAME\_DF_raw
#python train.py --detector_path $CONFIG_FILE  --train_dataset FF-FS --gpus ${GPUS[*]} --task_name $TASK_NAME\_FS_raw
python train.py --opts detector_path=$CONFIG_FILE  train_dataset=FaceForensics++ gpus=${GPUS[*]} task_name=$TASK_NAME\_all_c23 compression=$COMPRESSION