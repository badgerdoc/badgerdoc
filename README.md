# Setup

1. Create and activate python virtual environment.
2. Run `setup.sh` to install dependencies.
3. Run `download.sh` to get the minimal set of files required to run inference.

# Run pipeline

Run pipeline on single pdf document

`python pipeline.py full <path-to-pdf> <results-output-dir>`

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
