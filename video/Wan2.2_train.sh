accelerate launch --config_file accelerate_config_5B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/root/Projects/balls/Wan/Wan_test" \
  --trainable_models "dit" \
  --condition_list "image,position,normals,dist_to_sphere,dir_to_sphere" \
  --targets "sphere_0,sphere_1" \
  --save_steps 10000 \
  --num_steps 250000 \
  --validation_every_steps 1000  \
  --dataset_num_workers 4 \
  --train_ev \
  --shift 2 \
  --data_path "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_lighting_16sep_resized,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_camera_16sep_resized,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_camera_20sep_resized,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_big_15oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_15oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_camera_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_lighting_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_big_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_rand_outdoor_anim_lighting_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_rand_outdoor_anim_camera_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_rand_outdoor_anim_spheres_big_16oct,\
s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_rand_outdoor_anim_spheres_16oct"