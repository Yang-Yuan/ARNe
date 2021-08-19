#!/bin/bash

source arne-env/bin/activate
CONFIGFILE=./default_config.json python datasets/npz_to_pt.py