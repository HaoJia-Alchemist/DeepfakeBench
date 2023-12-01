cd /home/jh/disk/workspace/DeepfakeBench/training
GPUS='[3]'
CONFIG_FILE=./config/detector/efficient_only.yaml
TASK_NAME=efficientnetb4_only_myefficient
COMPRESSION=c23
train_batchSize=80
test_batchSize=140

python train.py --config_file $CONFIG_FILE  --opts train_dataset=[FF-NT] gpus=${GPUS[*]} task_name=$TASK_NAME\_NT_$COMPRESSION compression=$COMPRESSION test_dataset=[FF-DF,FF-F2F,FF-FS,FF-NT] train_batchSize=$train_batchSize test_batchSize=$test_batchSize

python train.py --config_file $CONFIG_FILE  --opts train_dataset=[FF-F2F] gpus=${GPUS[*]} task_name=$TASK_NAME\_F2F_$COMPRESSION compression=$COMPRESSION test_dataset=[FF-DF,FF-F2F,FF-FS,FF-NT] train_batchSize=$train_batchSize test_batchSize=$test_batchSize

python train.py --config_file $CONFIG_FILE  --opts train_dataset=[FF-DF] gpus=${GPUS[*]} task_name=$TASK_NAME\_DF_$COMPRESSION compression=$COMPRESSION test_dataset=[FF-DF,FF-F2F,FF-FS,FF-NT] train_batchSize=$train_batchSize test_batchSize=$test_batchSize

python train.py --config_file $CONFIG_FILE  --opts train_dataset=[FF-FS] gpus=${GPUS[*]} task_name=$TASK_NAME\_FS_$COMPRESSION compression=$COMPRESSION test_dataset=[FF-DF,FF-F2F,FF-FS,FF-NT] train_batchSize=$train_batchSize test_tchSize=$test_batchSize

python train.py  --config_file $CONFIG_FILE --opts train_dataset=[FaceForensics++] gpus=${GPUS[*]} task_name=$TASK_NAME\_FF_all_$COMPRESSION compression=$COMPRESSION test_dataset=[FaceForensics++,FF-DF,FF-F2F,FF-FS,FF-NT,Celeb-DF-v1,Celeb-DF-v2,DeepFakeDetection,FaceShifter,UADFV] train_batchSize=$train_batchSize test_batchSize=$test_batchSize