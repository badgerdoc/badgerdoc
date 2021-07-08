### Preprocess

To run preprocess for training use this:

```python -m training.preprocess <source pdfs and structure path> <output path> --verbose=True```

Source directory should contain two dirs ```pdfs``` and ```json```

**pdfs** should contain source pdfs for training

**json** should contain extracted structure with the same names as source pdfs (and json extension). 
Preprocess pipeline expects json in BadgerDoc returnable format. 
We expect that all BadgerDoc results will be processed manually to correct possible mistakes

```output dir``` will contain debug information for each pdf and **ttv** directory with prepared test train validation splits for training

! Before start inference check annotations in ```draw_ann``` directory for each pdf located in ```<output_dir>/pdfs/<pdf_name>/draw_ann```.
If any errors occurs - try to fix annotations in ttv directory with [coco annotator](https://github.com/jsbroks/coco-annotator) or any other annotation tool

### Training
! Training can be run only on the machine with GPU. Would be better to use at least Tesla T4.
####Docker preparation:
If you would like to store models on S3, Before running training setup following environment variables for example with docker env file
AWS_S3_ENDPOINT       # S3 endpoint url
AWS_ACCESS_KEY_ID     # aws access key id
AWS_SECRET_ACCESS_KEY # aws secret access key
AWS_REGION            # AWS region name
AWS_S3_SSE_TYPE       # AWS SSE type (optional)
####Run training:
To run training use with command: 
```
python -m training.train 
          <path-to-dataset> # after preprocess should be ttv dir
          <path-to-working-dir> # directory to store logs and trained models
          --load-from <path-to-base-model(optional)> # path to base models for training if not provided will be started training from scratch, 
                                                     # also AWS S3 path could be provided in format s3://<bucket_name>/<file_path>
          --resume-from <path-to-last-succesfull-epoch(optional)> # You could use it if training was interrupted for some reason - 
                                                                  # provide here last saved epoch from working dir, training will continue from it
          --config <path-to-configuration(optional)> # By default will be used configs/config_3_cls_w18.py, but you could redefine it and provide your config
                                                     # To understand config options please refere to https://github.com/open-mmlab/mmdetection
          --model_output <path-to-store-model-out>   # path where to save last trained model from working dir,
                                                     # also AWS S3 path could be provided in format s3://<bucket_name>/<file_path>
          --num-epoch <num-epochs-to-train>
          --demo <False/True>                        # To run demo mode
```

### Use trained models
To use trained models redefine environment variables ```CASCADE_CONFIG_PATH``` and ```CASCADE_MODEL_PATH``` before start

## Docker file
Also ```training/train.Docker``` file provided to train your models.
### Training docker requirements
* CUDA 11.0 (normally AWS instances with GPU already have CUDA drivers)
* nvidia-docker2 https://github.com/NVIDIA/nvidia-docker
* k8s-device-plugin https://github.com/NVIDIA/k8s-device-plugin

**To run your model training use ```python3``` instead of ```python```**