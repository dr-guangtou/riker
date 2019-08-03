#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get 1-D profiles from the maps."""

import numpy as np
import sep

def detect_galaxy(info, mass_map, kernel=None, threshold=1e8,
                  img_cen_x=None, img_cen_y=None, verbose=False,
                  bkg_ratio=10, bkg_filter=5):
    """Detect the galaxy and get the average shape of the isophote.

    Parameters
    ----------
    info: dict
        Basic information of the galaxy.
    map: ndarray
        Stellar mass map or 2-D map.
    kernel: ndarray, optional
        2-D kernel used for detecting the galaxy. Default: None
    threshold: float, optional
        Mass threshold for detecting the galaxy. Default: 1E8
    img_cen_x: float, optional
        X coordinate of the galaxy center. Default: None
    img_cen_y: float, optional
        Y coordinate of the galaxy center. Default: None
    bkg_ratio: int, optional
        Ratio between the image size and sky box size. Default: 10
    bkg_filter: int, optional
        Filter size for sky background. Default: 5
    verbose: bool, optional
        Blah, Blah, Blah. Default: False

    Returns
    -------
    detect: dict
        A dictionary that contains the object detection result.

    """
    # If no additional information is provided, assume the galaxy centers at
    # the image center
    if img_cen_x is None:
        img_cen_x = info['img_cen_x']
    if img_cen_y is None:
        img_cen_y = info['img_cen_y']

    # There is no "background", we do this to make sure the shape is
    # measured for the central region of the galaxy
    bkg = sep.Background(
        mass_map, bw=(info['img_w'] / bkg_ratio), bh=(info['img_h'] / bkg_ratio),
        fw=bkg_filter, fh=bkg_filter)

    objs = sep.extract(
        mass_map - bkg.back(), threshold, filter_type='conv',
        filter_kernel=kernel, segmentation_map=False)
    n_objs = len(objs)
    if verbose and n_objs > 1:
        print("# Detect {} objects on the map for galaxy {}".format(n_objs, info['id']))

    # Find the objsect at the center
    index = np.argmin(np.sqrt((objs['x'] - img_cen_x) ** 2.0 +
                              (objs['y'] - img_cen_y) ** 2.0))

    # Get the naive ba, theta, xcen, ycen
    ba = objs[index]['b'] / objs[index]['a']
    theta = objs[index]['theta']
    xcen, ycen = objs[index]['x'], objs[index]['y']

    return {'x': xcen, 'y': ycen, 'ba': ba, 'theta': theta,
            'pa': np.rad2deg(theta), 'objs': objs, 'n_objs': n_objs}
