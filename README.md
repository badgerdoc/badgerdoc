# Setup

1. Create and activate python virtual environment.
2. Run `setup.sh` to install dependencies.
3. Run `download.sh` to get the minimal set of files required to run inference.

# Run pipeline

Run pipeline on single pdf document

`python pipeline.py single <path-to-pdf> <results-output-dir>`

Results folder will have next structure:
```
├── sample_10.pdf                        # input pdf filename
    ├── epoch_36_mmd_v2_1607489706.json  # results of table extraction
    ├── images                           # pdf pages snapshots
    │    └── 0.png 
    ├── marked                           # pdf pages with tags
    │    └── 1607489706
    │           └── 0.png
    ├── ocr                              # tesseract results for each page
    │    └── 0.png.hocr
    └── text_data.json                   # text with coordinates extraction
```

# Inference

Run inference on single image and show the result.

`python inference.py show images/demo.png models/epoch_36_mmd_v2.pth models/cascadetabnet_config.py`


Run inference on batch of images and store results.

Use `--limit` to specify how many images from directory to process.

`python inference.py batch <img_dir> <out_dir> <model_file> <model_config> `

`python inference.py batch images output models/epoch_36_mmd_v2.pth models/cascadetabnet_config.py`


Draw tables on processed documents.

`python inference.py draw <img_dir> <out_dir> <prediction_json> `

`python inference.py draw images tables_over_images output/epoch_36_mmd_v2_1607473987.json `