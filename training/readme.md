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
! Training can be run only on the machine with GPU. Would be better to use at least Tesla K80 or Tesla T4. With Tesla T 4 you could achieve better performance

To run training use with command: 
```
python -m training.train 
          <path-to-dataset> # after preprocess should be ttv dir
          <path-to-working-dir> # directory to store logs and trained models
          --load-from <path-to-base-model(optional)> # path to base models for training if not provided will be started training from scratch
          --resume-from <path-to-last-succesfull-epoch(optional)> # You could use it if training was interrupted for some reason - 
                                                                  # provide here last saved epoch from working dir, training will continue from it
          --config <path-to-configuration(optional)> # By default will be used configs/config_3_cls_w18.py, but you could redefine it and provide your config
                                                     # To understand config options please refere to https://github.com/open-mmlab/mmdetection
          --num-epoch <num-epochs-to-train>
```

### Use trained models
To use trained models redefine environment variables ```CASCADE_CONFIG_PATH``` and ```CASCADE_MODEL_PATH``` before start