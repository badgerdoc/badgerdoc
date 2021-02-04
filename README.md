# Setup

1. Create and activate python virtual environment.
2. Run `setup.sh` to install dependencies.
3. Run `download.sh` to get the minimal set of files required to run inference.

# Run pipeline

Run pipeline on single pdf document

`python -m table_extractor.run run_sequentially <path-to-pdf> <results-output-dir> --verbose <true/false> --paddle_on <true/false>`

Results folder will have next structure:
