#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization of the results."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from matplotlib import rcParams

from kungpao.display import display_single

from riker.data import Z_SUN

plt.rc('text', usetex=True)
rcParams.update({'axes.linewidth': 1.5})
rcParams.update({'xtick.direction': 'in'})
rcParams.update({'ytick.direction': 'in'})
rcParams.update({'xtick.minor.visible': 'True'})
rcParams.update({'ytick.minor.visible': 'True'})
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '8.0'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '4.0'})
rcParams.update({'xtick.minor.width': '1.5'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '8.0'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '4.0'})
rcParams.update({'ytick.minor.width': '1.5'})
rcParams.update({'axes.titlepad': '10.0'})
rcParams.update({'font.size': 25})

__all__ = [
    'show_maps',
    'show_aper',
    'prepare_show_ellipse',
    'overplot_ellipse',
    'plot_ell_prof',
    'plot_ell_fourier',
]

# Color maps
IMG_CMAP = plt.get_cmap('Greys')
IMG_CMAP.set_bad(color='w')


def show_maps(maps, aper, cid=None, logms=None, figsize=(15, 15)):
    """Visualize the stellar mass, age, and metallicity maps.

    Parameters
    ----------
    maps : dict
        Dictionary that contains all stellar mass, age, and metallicity maps.
    aper : dict
        Dictionary that contains basic shape information of the galaxy.
    cid : int, optional
        `catsh_id`, sub-halo ID in the simulation. Used to identify galaxy.
        Default: None
    logms : float, optional
        Stellar mass in log10 unit. Default: None.
    figsize : tuple, optional
        Size of the 3x3 figure. Default: (15, 15)

    """
    # Setup the figure and grid of axes
    fig_sum = plt.figure(figsize=figsize, constrained_layout=False)
    grid_sum = fig_sum.add_gridspec(3, 3, wspace=0.0, hspace=0.0)
    fig_sum.subplots_adjust(
        left=0.005, right=0.995, bottom=0.005, top=0.995,
        wspace=0.00, hspace=0.00)

    # List of the maps need to be plot
    list_maps = ['mass_gal', 'mass_ins', 'mass_exs',
                 'age_gal', 'age_ins', 'age_exs',
                 'met_gal', 'met_ins', 'met_exs']

    for ii, name in enumerate(list_maps):
        ax = fig_sum.add_subplot(grid_sum[ii])
        if 'mass' in name:
            if ii % 3 == 0:
                _ = display_single(
                    maps[name], ax=ax, stretch='log10', zmin=6.0, zmax=10.5,
                    color_bar=True, scale_bar=False, no_negative=True,
                    color_bar_height='5%', color_bar_width='85%', color_bar_fontsize=20,
                    cmap=IMG_CMAP, color_bar_color='k')
                _ = ax.text(0.05, 0.06, r'$\log[M_{\star}/M_{\odot}]$', fontsize=25,
                            transform=ax.transAxes,
                            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))
            else:
                _ = display_single(
                    maps[name], ax=ax, stretch='log10', zmin=6.0, zmax=10.5,
                    color_bar=False, scale_bar=False, no_negative=True,
                    cmap=IMG_CMAP, color_bar_color='k')
            if ii == 0:
                # Label the center of the galaxy
                ax.scatter(aper['x'], aper['y'], marker='+', s=200,
                           c='orangered', linewidth=2.0, alpha=0.6)
                # Show the isophote shape
                e = Ellipse(xy=(aper['x'], aper['y']),
                            height=80.0 * aper['ba'], width=80.0, angle=aper['pa'])
                e.set_facecolor('none')
                e.set_edgecolor('orangered')
                e.set_alpha(0.5)
                e.set_linewidth(2.0)
                ax.add_artist(e)
                # Central
                ax.text(0.75, 0.06, r'$\rm Total$', fontsize=25,
                        transform=ax.transAxes)
            # Put the ID
            if ii == 1 and cid is not None and logms is not None:
                ax.text(
                    0.5, 0.88, r'$\mathrm{ID}: %d\ \ \log M_{\star}: %5.2f$' % (cid, logms),
                    fontsize=25, transform=ax.transAxes, horizontalalignment='center',
                    bbox=dict(facecolor='w', edgecolor='none', alpha=0.7))
                ax.text(0.75, 0.06, r'$\rm In\ situ$', fontsize=25,
                        transform=ax.transAxes)
            if ii == 2:
                ax.text(0.75, 0.06, r'$\rm Ex\ situ$', fontsize=25,
                        transform=ax.transAxes)
        if 'age' in name:
            if ii % 3 == 0:
                _ = display_single(
                    maps[name], ax=ax, stretch='linear', zmin=1.0, zmax=8.5,
                    color_bar=True, scale_bar=False, no_negative=True,
                    color_bar_height='5%', color_bar_width='85%', color_bar_fontsize=20,
                    cmap=IMG_CMAP, color_bar_color='k')
                _ = ax.text(0.06, 0.06, r'$\rm Age/Gyr$', fontsize=25,
                            transform=ax.transAxes,
                            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))
            else:
                _ = display_single(
                    maps[name], ax=ax, stretch='linear', zmin=1.0, zmax=8.5,
                    color_bar=False, scale_bar=False, no_negative=True,
                    cmap=IMG_CMAP, color_bar_color='k')
        if 'met' in name:
            if ii % 3 == 0:
                _ = display_single(
                    maps[name] / Z_SUN, ax=ax, stretch='log10', zmin=-0.6, zmax=0.9,
                    color_bar=True, scale_bar=False, no_negative=True,
                    color_bar_height='5%', color_bar_width='85%', color_bar_fontsize=20,
                    cmap=IMG_CMAP, color_bar_color='k')
                _ = ax.text(0.06, 0.06, r'$\log[Z_{\star}/Z_{\odot}]$', fontsize=25,
                            transform=ax.transAxes,
                            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))
            else:
                _ = display_single(
                    maps[name] / Z_SUN, ax=ax, stretch='log10', zmin=-0.6, zmax=0.9,
                    color_bar=False, scale_bar=False, no_negative=True,
                    cmap=IMG_CMAP, color_bar_color='k')

    return fig_sum


def show_aper(info, aper, figsize=(8, 18), rad_min=5.5, rad_max=170.):
    """Make a summary plot of the aperture measurements.

    Parameters
    ----------
    info : dict
        A dictionary that contains basic information of the galaxy
    aper : dict
        A dictionary that contains the aperture measurements of stellar mass, age,
        and metallicity.
    rad_min : float, optional
        Minimum radius to plot, in unit of kpc. Default: 5.5.
    rad_max : float, optional
        Maximum radius to plot, in unit of kpc. Default: 170.
    figsize : tuple, optional
        Size of the 3x3 figure. Default: (15, 15)

    """
    # Integrated properties of the galaxy
    logms, age = info['logms'], info['age']
    logz = np.log10(info['metallicity'] / Z_SUN)

    # Radial mask
    rad_mask = aper['rad_mid'] >= rad_min

    # Setup the figure
    fig_prof = plt.figure(figsize=figsize, constrained_layout=False)
    grid_prof = fig_prof.add_gridspec(4, 1, wspace=0.0, hspace=0.0)
    fig_prof.subplots_adjust(
        left=0.175, right=0.93, bottom=0.055, top=0.995,
        wspace=0.00, hspace=0.00)

    # Integrated mass profile
    ax0 = fig_prof.add_subplot(grid_prof[0])
    ax0.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['maper_gal']),
        c='darkgrey', marker='s', s=60, label=r'$\rm Total$')
    ax0.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['maper_ins']),
        c='orangered', marker='o', alpha=0.8, s=70, label=r'$\rm In\ situ$')
    ax0.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['maper_exs']),
        c='steelblue', marker='h', alpha=0.8, s=80, label=r'$\rm Ex\ situ$')
    ax0.axhline(logms, linewidth=2.5, linestyle='--', alpha=0.8, c='k',
                label='__no_label__')

    ax0.legend(fontsize=22, loc='best')
    ax0.grid(linestyle='--', alpha=0.5)

    _ = ax0.set_xlim(rad_min ** 0.25, rad_max ** 0.25)

    mass_arr = np.stack(
        [np.log10(aper['maper_gal'][rad_mask]),
         np.log10(aper['maper_ins'][rad_mask]),
         np.log10(aper['maper_exs'][rad_mask])])

    _ = ax0.set_ylim(np.nanmin(mass_arr) - 0.09, logms + 0.15)
    _ = ax0.set_ylabel(r'$\rm Curve\ of\ Growth$', fontsize=28)

    # Radial mass bin profile
    ax1 = fig_prof.add_subplot(grid_prof[1])
    ax1.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['mprof_ins']),
        c='orangered', marker='o', alpha=0.8, s=70, label=r'$\rm In\ situ$')
    ax1.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['mprof_exs']),
        c='steelblue', marker='h', alpha=0.8, s=80, label=r'$\rm Ex\ situ$')

    ax1.grid(linestyle='--', alpha=0.5)

    _ = ax1.set_xlim(rad_min ** 0.25, rad_max ** 0.25)

    mbins_arr = np.stack(
        [np.log10(aper['mprof_ins'][rad_mask]), np.log10(aper['mprof_exs'][rad_mask])])
    _ = ax1.set_ylim(np.nanmin(mbins_arr) * 0.95, np.nanmax(mbins_arr) * 1.05)
    _ = ax1.set_ylabel(r'$\log [M_{\star}/M_{\odot}]$', fontsize=28)

    # Ex-situ fraction
    fexs = aper['mprof_exs'] / (aper['mprof_ins'] + aper['mprof_exs'])

    ax1_b = fig_prof.add_axes(ax1.get_position())
    ax1_b.patch.set_visible(False)
    ax1_b.xaxis.set_visible(False)
    ax1_b.spines['right'].set_color('maroon')
    ax1_b.tick_params(axis='y', colors='maroon')
    ax1_b.yaxis.set_label_position('right')
    ax1_b.yaxis.set_ticks_position('right')
    ax1_b.plot(aper['rad_mid'] ** 0.25, fexs, linestyle='--', c='maroon',
               linewidth=3.5, alpha=0.8, label=r'$\rm Ex\ situ\ fraction$')
    ax1_b.set_ylim(0.02, 0.98)
    ax1_b.legend(fontsize=22, loc='best')

    # Metallicity profiles
    ax2 = fig_prof.add_subplot(grid_prof[2])
    ax2.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['met_gal_w'] / Z_SUN),
        c='darkgrey', marker='s', s=60, label='__no_label__')
    ax2.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['met_ins_w'] / Z_SUN),
        c='orangered', marker='o', s=70, alpha=0.8, label='__no_label__')
    ax2.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['met_exs_w'] / Z_SUN),
        c='steelblue', marker='h', s=80, alpha=0.8, label='__no_label__')
    ax2.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['met_ins'] / Z_SUN),
        edgecolor='orangered', marker='o', s=80, alpha=0.8, label='__no_label__',
        facecolor='none', linewidth=2)
    ax2.scatter(
        aper['rad_mid'] ** 0.25, np.log10(aper['met_exs'] / Z_SUN),
        edgecolor='steelblue', marker='h', s=90, alpha=0.8, label='__no_label__',
        facecolor='none', linewidth=2)
    ax2.axhline(logz, linewidth=2.5, linestyle='--', alpha=0.8, c='k',
                label=r'$\rm Catalog\ value$')

    ax2.grid(linestyle='--', alpha=0.5)
    ax2.legend(fontsize=22, loc='best')

    _ = ax2.set_xlim(rad_min ** 0.25, rad_max ** 0.25)
    met_arr = np.stack(
        [np.log10(aper['met_ins_w'][rad_mask] / Z_SUN),
         np.log10(aper['met_exs_w'][rad_mask] / Z_SUN)])
    _ = ax2.set_ylim(np.nanmin(met_arr) - 0.15, np.nanmax(met_arr) + 0.09)
    _ = ax2.set_ylabel(r'$\log [Z_{\star}/Z_{\odot}]$', fontsize=28)

    # Age profiles
    ax3 = fig_prof.add_subplot(grid_prof[3])
    ax3.scatter(
        aper['rad_mid'] ** 0.25, aper['age_gal_w'],
        c='darkgrey', marker='s', s=60, label='__no_label__')
    ax3.scatter(
        aper['rad_mid'] ** 0.25, aper['age_ins_w'],
        c='orangered', marker='o', s=70, alpha=0.8, label=r'$\rm Weighted$')
    ax3.scatter(
        aper['rad_mid'] ** 0.25, aper['age_exs_w'],
        c='steelblue', marker='h', s=80, alpha=0.8, label='__no_label__')
    ax3.scatter(
        aper['rad_mid'] ** 0.25, aper['age_ins'],
        edgecolor='orangered', marker='o', s=80, alpha=0.8, label=r'$\rm Not\ Weighted$',
        facecolor='none', linewidth=2)
    ax3.scatter(
        aper['rad_mid'] ** 0.25, aper['age_exs'],
        edgecolor='steelblue', marker='h', s=90, alpha=0.8, label='__no_label__',
        facecolor='none', linewidth=2)
    ax3.axhline(age, linewidth=2.5, linestyle='--', alpha=0.8, c='k',
                label=r'__no_label__')

    ax3.grid(linestyle='--', alpha=0.5)
    ax3.legend(fontsize=20, loc='best')

    _ = ax3.set_xlim(rad_min ** 0.25, rad_max ** 0.25)
    age_arr = np.stack(
        [aper['age_ins_w'][rad_mask], aper['age_exs_w'][rad_mask]])
    _ = ax3.set_ylim(np.nanmin(age_arr) * 0.8, np.nanmax(age_arr) + 1.5)

    _ = ax3.set_xlabel(r'$[R/{\rm kpc}]^{1/4}$', fontsize=28)
    _ = ax3.set_ylabel(r'$[\rm Age/Gyrs]$', fontsize=28)

    return fig_prof


def prepare_show_ellipse(summary, ellip):
    """Prepare the data for visualizing the 1-D profiles."""
    return {'catsh_id': summary['catsh_id'],
            'logms': summary['logms'],
            'mass_gal': summary['maps']['mass_gal'],
            'mass_ins': summary['maps']['mass_ins'],
            'mass_exs': summary['maps']['mass_exs'],
            'ell_gal_2': ellip['ell_gal_2'],
            'ell_gal_3': ellip['ell_gal_3'],
            'ell_ins_2': ellip['ell_ins_2'],
            'ell_ins_3': ellip['ell_ins_3'],
            'ell_exs_2': ellip['ell_exs_2'],
            'ell_exs_3': ellip['ell_exs_3']
           }


def overplot_ellipse(ell_plot, pix=1.0, zmin=3.5, zmax=10.5):
    """Overplot the elliptical isophotes on the stellar mass maps."""
    # Setup the figure
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0.005, right=0.995, bottom=0.005, top=0.995,
        wspace=0.00, hspace=0.00)

    # Build the grid
    gs = GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.00)

    # Central galaxy: step 2
    ax1 = fig.add_subplot(gs[0])
    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1 = display_single(
        ell_plot['mass_gal'], ax=ax1, stretch='log10', zmin=zmin, zmax=zmax,
        cmap=IMG_CMAP, no_negative=True, color_bar=True, scale_bar=False,
        color_bar_color='k')

    if ell_plot['ell_gal_2'] is not None:
        for k, iso in enumerate(ell_plot['ell_gal_2']):
            if k % 3 == 0 and iso['sma'] >= 6.0:
                e = Ellipse(xy=(iso['x0'], iso['y0']), height=iso['sma'] * 2.0,
                            width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                            angle=iso['pa'])
                e.set_facecolor('none')
                e.set_edgecolor('k')
                e.set_alpha(0.6)
                e.set_linewidth(2.0)
                ax1.add_artist(e)
        ax1.set_aspect('equal')

    _ = ax1.text(0.05, 0.06, r'$\rm Total$', fontsize=25,
                 transform=ax1.transAxes,
                 bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))

    # Central galaxy: step 3
    ax2 = fig.add_subplot(gs[1])
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2 = display_single(
        ell_plot['mass_gal'], ax=ax2, stretch='log10', zmin=zmin, zmax=zmax,
        cmap=IMG_CMAP, no_negative=True, color_bar=False, scale_bar=True,
        pixel_scale=1., physical_scale=pix, scale_bar_loc='right',
        scale_bar_length=50., scale_bar_color='k', scale_bar_y_offset=1.3)

    ax2.text(
        0.5, 0.92,
        r'$\mathrm{ID}: %d\ \ \log M_{\star}: %5.2f$' % (
            ell_plot['catsh_id'], ell_plot['logms']),
        fontsize=21, transform=ax2.transAxes,
        horizontalalignment='center', verticalalignment='center',
        bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))

    # Show the average isophotal shape
    if ell_plot['ell_ins_3'] is not None:
        n_iso = len(ell_plot['ell_ins_3'])
        if n_iso > 15:
            idx_use = n_iso - 6
        else:
            idx_use = n_iso - 1
        for k, iso in enumerate(ell_plot['ell_ins_3']):
            if k == idx_use:
                e = Ellipse(xy=(iso['x0'], iso['y0']), height=iso['sma'] * 2.0,
                            width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                            angle=iso['pa'])
                e.set_facecolor('none')
                e.set_edgecolor('k')
                e.set_linestyle('--')
                e.set_alpha(0.8)
                e.set_linewidth(2.5)
                ax2.add_artist(e)
        ax2.set_aspect('equal')

    _ = ax2.text(0.05, 0.06, r'$\rm Total$', fontsize=25,
                 transform=ax2.transAxes,
                 bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))

    # In situ component: step 2
    ax3 = fig.add_subplot(gs[2])
    ax3.yaxis.set_major_formatter(NullFormatter())
    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3 = display_single(
        ell_plot['mass_ins'], ax=ax3, stretch='log10', zmin=zmin, zmax=zmax,
        cmap=IMG_CMAP, no_negative=True, color_bar=False, scale_bar=False)

    if ell_plot['ell_ins_2'] is not None:
        for k, iso in enumerate(ell_plot['ell_ins_2']):
            if k % 3 == 0 and iso['sma'] >= 6.0:
                e = Ellipse(xy=(iso['x0'], iso['y0']), height=iso['sma'] * 2.0,
                            width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                            angle=iso['pa'])
                e.set_facecolor('none')
                e.set_edgecolor('orangered')
                e.set_alpha(0.9)
                e.set_linewidth(2.0)
                ax3.add_artist(e)
        ax3.set_aspect('equal')

    _ = ax3.text(0.05, 0.06, r'$\rm In\ situ$', fontsize=25,
                 transform=ax3.transAxes,
                 bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))

    # Ex situ component: step 2
    ax4 = fig.add_subplot(gs[3])
    ax4.yaxis.set_major_formatter(NullFormatter())
    ax4.xaxis.set_major_formatter(NullFormatter())
    ax4 = display_single(
        ell_plot['mass_exs'], ax=ax4, stretch='log10', zmin=zmin, zmax=zmax,
        cmap=IMG_CMAP, no_negative=True, color_bar=False, scale_bar=False)

    if ell_plot['ell_exs_2'] is not None:
        for k, iso in enumerate(ell_plot['ell_exs_2']):
            if k % 3 == 0 and iso['sma'] >= 6.0:
                e = Ellipse(xy=(iso['x0'], iso['y0']), height=iso['sma'] * 2.0,
                            width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                            angle=iso['pa'])
                e.set_facecolor('none')
                e.set_edgecolor('steelblue')
                e.set_alpha(0.9)
                e.set_linewidth(2.0)
                ax4.add_artist(e)
        ax4.set_aspect('equal')

    _ = ax4.text(0.05, 0.06, r'$\rm Ex\ situ$', fontsize=25,
                 transform=ax4.transAxes,
                 bbox=dict(facecolor='w', edgecolor='none', alpha=0.8))

    return fig


def plot_ell_prof(ell_plot, pix=1.0, r_min=3.0, r_max=190.0):
    """Plot a summary plot for the ellipse result."""
    # Setup the figure and axes
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0.0, right=1.0, bottom=0.00, top=1.0,
        wspace=0.00, hspace=0.00)

    ax1 = fig.add_axes([0.09, 0.10, 0.90, 0.48])
    ax2 = fig.add_axes([0.09, 0.58, 0.90, 0.21])
    ax3 = fig.add_axes([0.09, 0.79, 0.90, 0.20])

    # 1-D profile
    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ell_plot['ell_gal_3'] is not None:
        ax1.errorbar(
            (ell_plot['ell_gal_3']['sma'] * pix) ** 0.25,
            np.log10(ell_plot['ell_gal_3']['intens']),
            yerr=ell_plot['ell_gal_3']['sbp_err'], markersize=8,
            color='darkgrey', alpha=0.8, fmt='s', capsize=3,
            capthick=1, elinewidth=1, label=r'$\mathrm{Total}$')

    if ell_plot['ell_ins_3'] is not None:
        ax1.errorbar(
            (ell_plot['ell_ins_3']['sma'] * pix) ** 0.25 + 0.05,
            np.log10(ell_plot['ell_ins_3']['intens']),
            yerr=ell_plot['ell_ins_3']['sbp_err'], markersize=9,
            color='orangered', alpha=0.8, fmt='o', capsize=3,
            capthick=1, elinewidth=1, label=r'$\mathrm{In\ Situ}$')

    if ell_plot['ell_exs_3'] is not None:
        ax1.errorbar(
            (ell_plot['ell_exs_3']['sma'] * pix) ** 0.25 - 0.05,
            np.log10(ell_plot['ell_exs_3']['intens']),
            yerr=ell_plot['ell_exs_3']['sbp_err'], markersize=9,
            color='steelblue', alpha=0.8, fmt='h', capsize=3,
            capthick=1, elinewidth=1, label=r'$\mathrm{Ex\ Situ}$')

    ax1.legend(loc='best', fontsize=23)

    if (ell_plot['ell_gal_3'] is not None and ell_plot['ell_ins_3'] is not None and
            ell_plot['ell_exs_3'] is not None):
        mass_arr = (
            list(ell_plot['ell_exs_3']['intens'][ell_plot['ell_exs_3']['sma'] * pix > r_min]) +
            list(ell_plot['ell_ins_3']['intens'][ell_plot['ell_ins_3']['sma'] * pix > r_min]) +
            list(ell_plot['ell_gal_3']['intens'][ell_plot['ell_gal_3']['sma'] * pix > r_min]))
        min_mass = np.nanmin(mass_arr)
        if min_mass > 100.0:
            ax1.set_ylim(np.log10(min_mass) - 0.3, np.log10(np.nanmax(mass_arr)) + 0.5)
        else:
            ax1.set_ylim(2.01, np.log10(np.nanmax(mass_arr)) + 0.5)

    ax1.set_xlim(r_min ** 0.25, r_max ** 0.25)

    _ = ax1.set_xlabel(r'$R/\mathrm{kpc}^{1/4}$', fontsize=28)
    _ = ax1.set_ylabel(r'$\log\ (\mu_{\star}/[M_{\odot}\ \mathrm{kpc}^{-2}])$', fontsize=28)

    # Ellipticity profile
    ax2.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ell_plot['ell_gal_3'] is not None:
        ax2.axhline(ell_plot['ell_gal_3']['ell'][1], c='k', linestyle='--',
                    linewidth=3, alpha=0.5)

    if ell_plot['ell_gal_2'] is not None:
        ax2.errorbar(
            (ell_plot['ell_gal_2']['sma'] * pix) ** 0.25, ell_plot['ell_gal_2']['ell'],
            yerr=ell_plot['ell_gal_2']['ell_err'], color='darkgrey', alpha=0.7, fmt='s',
            capsize=3, capthick=1, elinewidth=2, markersize=8)

    if ell_plot['ell_ins_2'] is not None:
        ax2.errorbar(
            (ell_plot['ell_ins_2']['sma'] * pix) ** 0.25 + 0.05, ell_plot['ell_ins_2']['ell'],
            yerr=ell_plot['ell_ins_2']['ell_err'], color='orangered', alpha=0.7, fmt='o',
            capsize=3, capthick=1, elinewidth=1, markersize=9)

    if ell_plot['ell_exs_2'] is not None:
        ax2.errorbar(
            (ell_plot['ell_exs_2']['sma'] * pix) ** 0.25 - 0.05, ell_plot['ell_exs_2']['ell'],
            yerr=ell_plot['ell_exs_2']['ell_err'], color='steelblue', alpha=0.6, fmt='h',
            capsize=3, capthick=1, elinewidth=1, markersize=9)

    if ell_plot['ell_exs_2'] is not None and ell_plot['ell_ins_2'] is not None:
        ell_arr = (
            list(ell_plot['ell_exs_2']['ell'][ell_plot['ell_exs_2']['sma'] * pix > r_min]) +
            list(ell_plot['ell_ins_2']['ell'][ell_plot['ell_ins_2']['sma'] * pix > r_min]))
        ax2.set_ylim(np.nanmin(ell_arr) - 0.05, np.nanmax(ell_arr) + 0.05)

    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.set_xlim(r_min ** 0.25, r_max ** 0.25)
    _ = ax2.set_ylabel(r'$e$', fontsize=25)

    # Position Angle profile
    ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ell_plot['ell_gal_3'] is not None:
        ax3.axhline(ell_plot['ell_gal_3']['pa'][1], c='k', linestyle='--',
                    linewidth=3, alpha=0.5)

    if ell_plot['ell_gal_2'] is not None:
        ax3.errorbar(
            (ell_plot['ell_gal_2']['sma'] * pix) ** 0.25, ell_plot['ell_gal_2']['pa'],
            yerr=ell_plot['ell_gal_2']['pa_err'], color='darkgrey', alpha=0.7, fmt='s',
            capsize=3, capthick=1, elinewidth=2, markersize=8)

    if ell_plot['ell_ins_2'] is not None:
        ax3.errorbar(
            (ell_plot['ell_ins_2']['sma'] * pix) ** 0.25 + 0.05, ell_plot['ell_ins_2']['pa'],
            yerr=ell_plot['ell_ins_2']['pa_err'], color='orangered', alpha=0.7, fmt='o',
            capsize=3, capthick=1, elinewidth=1, markersize=9)

    if ell_plot['ell_exs_2'] is not None:
        ax3.errorbar(
            (ell_plot['ell_exs_2']['sma'] * pix) ** 0.25 - 0.05, ell_plot['ell_exs_2']['pa'],
            yerr=ell_plot['ell_exs_2']['pa_err'], color='steelblue', alpha=0.6, fmt='h',
            capsize=3, capthick=1, elinewidth=1, markersize=9)

    if ell_plot['ell_exs_2'] is not None and ell_plot['ell_ins_2'] is not None:
        pa_arr = (
            list(ell_plot['ell_exs_2']['pa'][ell_plot['ell_exs_2']['sma'] * pix > r_min]) +
            list(ell_plot['ell_ins_2']['pa'][ell_plot['ell_ins_2']['sma'] * pix > r_min]))
        ax3.set_ylim(np.nanmin(pa_arr) - 10.0, np.nanmax(pa_arr) + 10.0)

    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.set_xlim(r_min ** 0.25, r_max ** 0.25)
    _ = ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=20)

    return fig


def plot_ell_fourier(fourier, pix=1.0, r_min=6.0, r_max=190.0, show_both=False):
    """Plot a summary plot for the ellipse result."""
    # Setup the figure and axes
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0.0, right=1.0, bottom=0.00, top=1.0,
        wspace=0.00, hspace=0.00)

    ax1 = fig.add_axes([0.13, 0.091, 0.865, 0.224])
    ax2 = fig.add_axes([0.13, 0.315, 0.865, 0.225])
    ax3 = fig.add_axes([0.13, 0.540, 0.865, 0.225])
    ax4 = fig.add_axes([0.13, 0.765, 0.865, 0.224])

    # A1 and/or B1
    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax1.axhline(0.0, linestyle='--', alpha=0.7, linewidth=2, color='k')

    ax1.errorbar(
        (fourier['r_pix'] * pix) ** 0.25, fourier['a1'],
        yerr=fourier['a1_err'], markersize=8,
        color='coral', alpha=0.9, fmt='o', capsize=3, capthick=2, elinewidth=2,
        label=r'$\mathrm{a}$')

    if show_both:
        ax1.errorbar(
            (fourier['r_pix'] * pix) ** 0.25, fourier['b1'],
            yerr=fourier['b1_err'], markersize=8,
            color='dodgerblue', alpha=0.6, fmt='s', capsize=3, capthick=2, elinewidth=2,
            label=r'$\mathrm{b}$')

        ax1.legend(loc='best', fontsize=15)

    ax1.set_xlim(r_min ** 0.25, r_max ** 0.25)
    ax1.yaxis.set_major_formatter(FormatStrFormatter(r'$%4.1f$'))

    _ = ax1.set_xlabel(r'$R/\mathrm{kpc}^{1/4}$', fontsize=28)
    if not show_both:
        _ = ax1.set_ylabel(r'$\rm a_{1}$', fontsize=30)
    else:
        _ = ax1.set_ylabel(r'$\rm a_{1}\ or\ b_{1}$', fontsize=30)

    # A2 and/or B2
    ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax2.axhline(0.0, linestyle='--', alpha=0.7, linewidth=2)

    ax2.errorbar(
        (fourier['r_pix'] * pix) ** 0.25, fourier['a2'],
        yerr=fourier['a2_err'], markersize=8,
        color='coral', alpha=0.9, fmt='o', capsize=3, capthick=2, elinewidth=2,
        label=r'$\mathrm{a2}$')

    if show_both:
        ax2.errorbar(
            (fourier['r_pix'] * pix) ** 0.25, fourier['b2'],
            yerr=fourier['b2_err'], markersize=8,
            color='dodgerblue', alpha=0.6, fmt='s', capsize=3, capthick=2, elinewidth=2,
            label=r'$\mathrm{b2}$')

    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.yaxis.set_major_formatter(FormatStrFormatter(r'$%4.1f$'))
    ax2.set_xlim(r_min ** 0.25, r_max ** 0.25)

    if not show_both:
        _ = ax2.set_ylabel(r'$\rm a_{2}$', fontsize=30)
    else:
        _ = ax2.set_ylabel(r'$\rm a_{2}\ or\ b_{2}$', fontsize=30)

    # A3 and/or B3
    ax3.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax3.axhline(0.0, linestyle='--', alpha=0.7, linewidth=2)

    ax3.errorbar(
        (fourier['r_pix'] * pix) ** 0.25, fourier['a3'],
        yerr=fourier['a3_err'], markersize=8,
        color='coral', alpha=0.9, fmt='o', capsize=3, capthick=2, elinewidth=2,
        label=r'$\mathrm{a}$')

    if show_both:
        ax3.errorbar(
            (fourier['r_pix'] * pix) ** 0.25, fourier['b3'],
            yerr=fourier['b3_err'], markersize=8,
            color='dodgerblue', alpha=0.6, fmt='s', capsize=3, capthick=2, elinewidth=2,
            label=r'$\mathrm{b}$')

    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.yaxis.set_major_formatter(FormatStrFormatter(r'$%4.1f$'))
    ax3.set_xlim(r_min ** 0.25, r_max ** 0.25)

    if not show_both:
        _ = ax3.set_ylabel(r'$\rm a_{3}$', fontsize=30)
    else:
        _ = ax3.set_ylabel(r'$\rm a_{3}\ or\ b_{3}$', fontsize=30)

    # A4 and/or B4
    ax4.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax4.axhline(0.0, linestyle='--', alpha=0.7, linewidth=2)

    ax4.errorbar(
        (fourier['r_pix'] * pix) ** 0.25, fourier['a4'],
        yerr=fourier['a4_err'], markersize=8,
        color='coral', alpha=0.9, fmt='o', capsize=3, capthick=2, elinewidth=2,
        label=r'$\mathrm{a}$')

    if show_both:
        ax4.errorbar(
            (fourier['r_pix'] * pix) ** 0.25, fourier['b4'],
            yerr=fourier['b4_err'], markersize=8,
            color='dodgerblue', alpha=0.6, fmt='s', capsize=3, capthick=2, elinewidth=2,
            label=r'$\mathrm{b}$')

    ax4.xaxis.set_major_formatter(NullFormatter())
    ax4.yaxis.set_major_formatter(FormatStrFormatter(r'$%4.1f$'))
    ax4.set_xlim(r_min ** 0.25, r_max ** 0.25)

    if not show_both:
        _ = ax4.set_ylabel(r'$\rm a_{4}$', fontsize=30)
    else:
        _ = ax4.set_ylabel(r'$\rm a_{4}\ or\ b_{4}$', fontsize=30)

    return fig
