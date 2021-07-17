# Setup

If you are using MacOS, then run `brew install poppler tesseract` first.

1. Create and activate python virtual environment.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Run `download.sh` to get the minimal set of files required to run inference.

# Docker setup
1. Run `docker build . -t badgerdoc-tb-extr`
2. Use built image for example like this `docker run -it badgerdoc-tb-extr bash`

Or any other way

# Use models from AWS S3
If you would like to download models from S3, Before running setup following environment variables for example with docker env file
AWS_S3_ENDPOINT       # S3 endpoint url
AWS_ACCESS_KEY_ID     # aws access key id
AWS_SECRET_ACCESS_KEY # aws secret access key
AWS_REGION            # AWS region name
AWS_S3_SSE_TYPE       # AWS SSE type (optional)

# Run excel or pdf pipeline

`python -m table_extractor.run run <path-to-pdf-or-excel> <results-output-dir> --model_path <model-file-path> --verbose <true/false> --paddle_on <true/false>`
Pipeline will automatically decide how parse your document based on file extension.
Supported file formats:
*.pdf, *.xlsx, *.xlsm, *.xltx, *.xltm

Model on S3 could be used if `<model-file-path>` provided in format `s3://<bucket>/<models_path>`

# Run pdf pipeline

Run pipeline on single pdf document

`python -m table_extractor.run run-sequentially <path-to-pdf> <results-output-dir> --model_path <model-file-path> --verbose <true/false> --paddle_on <true/false>`

Model on S3 could be used if `<model-file-path>` provided in format `s3://<bucket>/<models_path>`

# Run excel extractor
`python -m excel_extractor.excel_run  <path-to-excel> <output-path>`
