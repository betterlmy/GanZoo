## ILVR Sampling
First, set PYTHONPATH variable to point to the root of the repository.
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
conda activate torch_diff
```
## Train
I used this command for training:
修改wight_dir（保存权重文件地址）和data_dir
```
nohup python scripts/image_train.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 2 --num_head_channels 128 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_dir models/cond+adpattn --data_dir datasets/CTLDR/train/ >log/cond+adpattn.out &
```
## sampling
And this for sampling:
修改model_path、base_samples、save_dir
```
nohup python scripts/ilvr_sample.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.2 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 128 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --down_N 4 --range_t 0 --use_ddim False --model_path models/CAWM/ema_0.9999_130000.pt --base_samples ref_imgs/CTLDR/quarter/L067/ --save_dir result/newmodel670000 > log/1.out &
```

## 继续上次的断点训练
修改其中的data_dir 和 resume_checkpoint和wight_dir（保存权重文件地址）
```
nohup python scripts/image_train.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.3 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 2 --num_head_channels 128 --resblock_updown True --use_fp16 False --use_scale_shift_norm True  --weight_dir models/cond+adpattn+CAWM/ --data_dir datasets/CTLDR/train --resume_checkpoint models/CAWM/model130000.pt >log/cond+adpattn+CAWM.out &
```
