# Project: Toxic Comment Classification

## Installation

Make sure you setup your virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -U -r requirements.txt

## Create output.zip

To create the `output.zip` file for upload to Coursys do:

    python3 zipout.py

For more options:

    python3 zipout.py -h

## Create source.zip

To create the `source.zip` file for upload to Coursys do:

    python3 zipsrc.py



## Data files

The data files provided are:

* `data/input` -- input files `dev.tsv` and `test.tsv`
* `data/reference/dev.out` -- the reference output for the `dev.tsv` input file

