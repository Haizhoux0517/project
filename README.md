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

For more options:

    python3 zipsrc.py -h

## Check your accuracy

To check your accuracy on the dev set:

    python3 check.py

For more options:

    python3 check.py -h

In particular use the log file to check your output evaluation:

    python3 check.py -l log

The accuracy on `data/input/test.tsv` will not be shown.  We will
evaluate your output on the test input after the submission deadline.

## Data files

The data files provided are:

* `data/input` -- input files `dev.tsv` and `test.tsv`
* `data/reference/dev.out` -- the reference output for the `dev.tsv` input file

