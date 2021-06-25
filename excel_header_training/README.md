# Prepare headers dictionary

Here is described how to prepare headers dictionary to improve headers recognition

## Preparing excel markup

Before running header extraction script, please prepare excel annotations.

Please provide next steps to prepare excel annotations:
1. Collect all excel files in xlsx format in one directory <path-to-excels-folder>.
2. Open each file and set background color for each header cell as '2979ff' (in hex) or any other, which you would like to use to mark headers.

## Extract dictionaries

1. Run extract header script:
```
python -m excel_header_training.extract_header \
    <path-to-excels-folder> # Path to directory containing annotated excels
    <path-to-output-folder> # Path to output folder, where result dictionaries will be stored
    --header_color          # Header background color, optional, by default will be used FF0066CC
    --header_excel          # Excel file with annotated headers, which could be used to take header color
    --header_sheet          # Sheet name in provided by --header_excel excel file, which could be used to take header color
    --header_cell           # Cell coordinate (in Excel format, like A12) on provided by --header_sheet sheet, which is header cell
```
2. Results will be stored in <path-to-output-folder> and will contain two files: ```headers.json``` and ```cells.json```
3. To use collected dictionaries setup next env variables before using excel extractor:
```
export CELL_DICT=<path-to-output-folder>/cells.json
export HEADER_DICT=<path-to-output-folder>/headers.json
```