# Setup

If you are using MacOS, then run `brew install poppler tesseract` first.

1. Create and activate python virtual environment.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Run `download.sh` to get the minimal set of files required to run inference.

# Run pipeline

Run pipeline on single pdf document

`python -m table_extractor.run run-sequentially <path-to-pdf> <results-output-dir> --verbose <true/false> --paddle_on <true/false>`

Results folder will have next structure:

# Run excel extractor
`python -m table_extractor.excel_run  <path-to-excel> <output-path>`
Note! To run excel extraction correctly, please set File -> Export as PDF -> Structure -> Whole Sheet Export
