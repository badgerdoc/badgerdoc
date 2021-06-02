To run preprocess for training use this:

```python -m training.preprocess <source pdfs and structure path> <output path> --verbose=True```

Source directory should contain two dirs ```pdfs``` and ```json```

**pdfs** should contain source pdfs for training

**json** should contain extracted structure with the same names as source pdfs (and json extension). 
Preprocess pipeline expects json in BadgerDoc returnable format. 
We expect that all BadgerDoc results will be processed manually to correct possible mistakes

```output dir``` will contain debug information for each pdf and **ttv** directory with prepared test train validation splits for training

