#!/usr/bin/env python
# energy
# Small app to compute energy efficiency of a house
#
# Author:   Benjamin Bengfort <benjamin.bengfort@georgetown.edu>
# Created:  Fri Mar 27 12:02:19 2015 -0400
#
# Copyright (C) 2015 Georgetown University
# For license information, see LICENSE.txt
#
# ID: energy.py [] benjamin.bengfort@georgetown.edu $

"""
Small app to compute energy efficiency of a house
"""

##########################################################################
## Imports
##########################################################################

import os
import time
import pickle
import argparse

from utils import load_energy

##########################################################################
## Module Constants
##########################################################################

PROGRAM = {
    'description': 'A simple energy efficiency computer',
    'epilog': 'Report bugs on Github',
    'version': 'energy v1.0'
}

# expects models to be in same directory
MODEL_DIR  = os.path.dirname(__file__)
HEAT_MODEL = os.path.join(MODEL_DIR, "heating.pickle")
COLD_MODEL = os.path.join(MODEL_DIR, "cooling.pickle")

##########################################################################
## Phases
##########################################################################


def build(args):
    """
    Builds the models from the arguments.
    In a real applciation, would probably arguments:

        - fixtures (where the training data is)
        - model_dir (where to write the models out to)
        - kfolds (number of cross validation folds)

    For now, just write out the pickles to HEAT_MODEL and COLD_MODEL
    """
    start = time.time()
    print dict(vars(args))
    build_time = time.time() - start

    return "Build took %0.3f seconds" % build_time


def predict(args):
    """
    Makes a prediction based on the models.

    You're going to have to load the pickles from disk, if they don't exist,
    you'll have to throw an error to tell the user to build them.

    Collect the arguments from the command line, then return two predictions
    using the heating and cooling models. Note - what if the user doesn't
    give all the values?

    Finally, collect feedback from the user to update the model.
    """
    print dict(vars(args))

    heating_load = 15.1  # use heat_model.predict()
    cooling_load = 15.1  # use cool_model.predict()

    return "Heating Load: %0.3f, Cooling Load: %0.3f" % (heating_load, cooling_load)

##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(**PROGRAM)
    subparsers = parser.add_subparsers(title='commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='train models')
    build_parser.set_defaults(func=build)

    # Predict command
    predict_arguments = (
        (('--compactness',), {'type': float}),
        (('--surface-area',), {'type': float}),
        (('--wall-area',), {'type': float}),
        (('--roof-area',), {'type': float}),
        (('--height',), {'type': float}),
        (('--orientation',), {'type': float}),
        (('--glazing-area',), {'type': float}),
        (('--glazing-distribution',), {'type': float}),
    )

    pred_parser  = subparsers.add_parser('predict', help='estimate efficiency')
    for args, kwargs in predict_arguments:
        pred_parser.add_argument(*args, **kwargs)
    pred_parser.set_defaults(func=predict)

    # Handle input from the command line
    args = parser.parse_args()              # Parse the arguments
    try:
        msg = args.func(args)               # Call the default function
        parser.exit(0, msg + "\n")          # Exit cleanly with message
    except Exception as e:
        parser.error(str(e))                # Exit with error
