GPUS=(1 2)
python train.py --detector_path ./config/detector/rgbmsnlc.yaml --train_dataset FF-F2F --gpus ${GPUS[*]} --task_name rgbmsnlc_F2F_raw
python train.py --detector_path ./config/detector/rgbmsnlc.yaml --train_dataset FF-DF --gpus ${GPUS[*]} --task_name rgbmsnlc_DF_raw
python train.py --detector_path ./config/detector/rgbmsnlc.yaml --train_dataset FF-FS --gpus ${GPUS[*]} --task_name rgbmsnlc_FS_raw