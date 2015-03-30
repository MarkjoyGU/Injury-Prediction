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
import numpy as np

from utils import load_energy
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score as cvs

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

    # Load data and estimator
    dataset   = load_energy()
    alphas    = np.logspace(-10, -2, 200)

    scores    = {}
    for y in ('Y1', 'Y2'):
        # Perform cross validation, don't worry about Imputation here
        clf = linear_model.RidgeCV(alphas=alphas)
        scores[y] = cvs(clf, dataset.data, dataset.target(y), cv=12)

        # Get the alpha from the ridge by fitting the entire data set.
        # There are a couple of reasons for this, but mostly to ensure that
        # we get the desired result pickled (e.g. a ridge with alpha)
        clf.fit(dataset.data, dataset.target(y))

        # Build the model on the entire datset include Imputer pipeline
        model     = linear_model.Ridge(alpha=clf.alpha_)
        imputer   = Imputer(missing_values="NaN", strategy="mean", axis=0)
        estimator = Pipeline([("imputer", imputer), ("ridge", model)])
        estimator.fit(dataset.data, dataset.target(y))

        # Dump the model
        jump  = {
            'Y1': HEAT_MODEL,
            'Y2': COLD_MODEL,
        }

        with open(jump[y], 'w') as f:
            pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)

        msg = (
            "%s trained on %i instances using a %s model\n"
            "     average R2 score of %0.3f using an alpha of %0.5f\n"
            "     model has been dumped to %s\n"
        )

        print msg % (
            y, len(dataset.data), model.__class__.__name__,
            scores[y].mean(), clf.alpha_,
            jump[y],
        )

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

    # Gather data from the command line (already floats or None)
    # Explicitly gathered for both documentation and correct order
    x = np.array([
        args.compactness,
        args.surface_area,
        args.wall_area,
        args.roof_area,
        args.height,
        args.orientation,
        args.glazing_area,
        args.glazing_distribution,
    ], dtype=np.float)

    # Show inputs to the user
    print "Predicting a value for the input vector:"
    print "    %s" % str(x)
    print "Note that nan values will be replaced by means\n"

    with open(HEAT_MODEL, 'rb') as hp:
        heat_model = pickle.load(hp)

    with open(COLD_MODEL, 'rb') as cp:
        cold_model = pickle.load(cp)

    heating_load = heat_model.predict(x)
    cooling_load = cold_model.predict(x)

    output = "Heating Load: %0.3f, Cooling Load: %0.3f"
    return output % (heating_load, cooling_load)

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
    # try:
    msg = args.func(args)               # Call the default function
    parser.exit(0, msg + "\n")          # Exit cleanly with message
    # except Exception as e:
    #     parser.error(str(e))                # Exit with error
