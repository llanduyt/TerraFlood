# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for thresholding
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit


def apply_ki(image, accuracy=200, fig_filename=None, t_min=-50.0, t_max=5.0):
    """Select threshold according to Kittler & Illingworth
    
    Inputs:
    image: nd array
        Array of pixel values.
    accuracy: int(default=200)
        Number of bins to construct histogram.
    fig_filename: str or None (default=None)
        If not None, a plot of the cost function is saved to the specified path
    Outputs:
    t: float
        Threshold value
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # Histogram    
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    g_pos = g - np.min(g)
    g01 = g_pos / np.max(g_pos)
    
    # Cost function and threshold
    c = np.cumsum(h)
    m = np.cumsum(h * g01)
    s = np.cumsum(h * g01**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    c[c == 0] = 1e-9
    cb[cb == 0] = 1e-9
    var_f = s/c - (m/c)**2
    if np.any(var_f < 0):
        var_f[var_f < 0] = 0
    sigma_f = np.sqrt(var_f)
    var_b = sb/cb - (mb/cb)**2
    if np.any(var_b < 0):
        var_b[var_b < 0] = 0
    sigma_b = np.sqrt(var_b)
    p = c / c[-1]
    sigma_f[sigma_f == 0] = 1e-9
    sigma_b[sigma_b == 0] = 1e-9
    j = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p+1e-9)
    j[~np.isfinite(j)] = np.nan
    g_v = np.array(find_valley(g, j))
    g_v = g_v[(g_v > t_min) & (g_v < t_max)]
    if len(g_v) == 0:
        t = np.nan
    elif len(g_v) == 1:
        t = g_v[0]
    else:
        g_v_j = j[np.isin(g, g_v)]
        t = g_v[np.argmin(g_v_j)]
    # Plot
    if fig_filename:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(g, j, color='k')
        ax[0].plot([t, t], [np.nanmin(j), np.nanmax(j)], 'r')
        ax[1].bar(g, h)
        ax[1].plot([t, t], [0, np.nanmax(h)], 'r')
        plt.savefig(fig_filename)
        plt.close(fig)
    # Return
    return t


def apply_otsu(image, accuracy=200, fig_filename=None):
    """Select threshold according to Otsu
    
    Inputs:
    image: nd array
        Array of pixel values.
    accuracy: int(default=200)
        Number of bins to construct histogram.
    fig_filename: str or None (default=None)
        If not None, a plot of the between-class variance is saved to the specified path
    Outputs:
    t: float
        Threshold value
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # Histogram
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    # Between class variance and threshold
    w1 = np.cumsum(h)
    w2 = w1[-1] - w1
    w2[w2 == 0] = 1e-9
    gh = np.cumsum(g*h)
    mu1 = gh/w1
    mu2 = (gh[-1]-gh)/w2
    var_between = w1*w2*(mu1-mu2)**2
    idx = np.nanargmax(var_between)
    t = g[idx]
    # Plot
    if fig_filename:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(g, var_between, color='k')
        ax[0].plot([t, t], [ax[0].get_ylim()[0], np.nanmax(var_between)], 'r')
        ax[1].bar(g, h)
        ax[1].plot([t, t], [0, np.nanmax(h)], 'r')
        plt.savefig(fig_filename)
        plt.close(fig)
    # Return
    return t


def tile_vars(image, selection='Martinis', t_method=['KI', 'Otsu'], tile_dim=[200, 200], hand_matrix=None,
              directory_figures=None, accuracy=200, incomplete_tile_warning=True):
    """ Calculate tile variables
    
    Inputs:
    image: nd array
        Array of pixel values.
    selection: str (default='Martinis')
        Method for tile selection. Currently only option is 'Martinis'.
    t_method: list (default=['KI', 'Otsu'])
        List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    hand_matrix: ndarray or None (default=None)
        Array of HAND values.
    incomplete_tile_warning: bool (default=True)
        Whether to give a warning when incomplete tiles are encountered.
    Outputs:
    tile_ki: nd array
        Array of KI thresholds.
    tile_o: nd array
        Array of Otsu thresholds.
    average: nd array
        Array of tile averages.
    stdev: nd array
        Array of tiled st. devs.
    hand: nd array
        Array of tile mean HAND values.
    """
    tile_rows, tile_cols = tile_dim
    nrt = np.ceil(image.shape[0]/tile_rows).astype('int')
    nct = np.ceil(image.shape[1]/tile_cols).astype('int')
    if selection == 'Martinis':
        stdev = np.full([nrt, nct], np.nan)
        average = np.full([nrt, nct], np.nan)
    elif selection == 'Chini':
        a = np.full([nrt, nct], np.nan)
        bc = np.full([nrt, nct], np.nan)
        sr = np.full([nrt, nct], np.nan)
    if hand_matrix is not None:
        hand = np.full([nrt, nct], np.nan)
    else:
        hand = None
    for r in np.arange(0, image.shape[0], tile_rows):
        tile_rindex = np.floor(r/tile_rows).astype('int')
        for c in np.arange(0, image.shape[1], tile_cols):
            tile_cindex = np.floor(c/tile_cols).astype('int')
            tile = image[r:min(r+tile_rows, image.shape[0]), c:min(c+tile_cols, image.shape[1])]
            if np.sum(np.isnan(tile)) <= 0.1*np.size(tile):
                if selection == 'Martinis':
                    tr, tc = tile.shape
                    mu1 = np.nanmean(tile[0:tr//2, 0:tc//2])
                    mu2 = np.nanmean(tile[0:tr//2, tc//2:])
                    mu3 = np.nanmean(tile[tr//2:, 0:tc//2])
                    mu4 = np.nanmean(tile[tr//2:, tc//2:])
                    stdev[tile_rindex, tile_cindex] = np.std([mu1, mu2, mu3, mu4]) 
                    average[tile_rindex, tile_cindex] = np.nanmean(tile)
                elif selection == 'Chini':
                    if tile[np.isnan(tile)].size / tile.size < 0.1:
                        if directory_figures:
                            fig_filename = os.path.join(directory_figures, "Tile{}-{}_HistFitting.png".format(
                                tile_rindex, tile_cindex))
                        else:
                            fig_filename = None
                        try:
                            x, y = get_hist(tile[~np.isnan(tile)], accuracy=accuracy)
                            par_dist = fit_two_gaussians(x, y, apply_otsu(tile, accuracy=accuracy),
                                                         fig_filename=fig_filename)
                            a[tile_rindex, tile_cindex] = get_ashman(par_dist[1], par_dist[2], par_dist[4], par_dist[5])
                            bc[tile_rindex, tile_cindex] = get_bc(x, y, par_dist)
                            sr[tile_rindex, tile_cindex] = get_sr(*par_dist)
                        except:
                            a[tile_rindex, tile_cindex] = np.nan
                            bc[tile_rindex, tile_cindex] = np.nan
                            sr[tile_rindex, tile_cindex] = np.nan
                if hand_matrix is not None:
                    hand[tile_rindex, tile_cindex] = np.nanmean(hand_matrix[r:min(r+tile_rows, image.shape[0]),
                                                                c:min(c+tile_cols, image.shape[1])])
            elif incomplete_tile_warning:
                print("Tile ({0:.0f}, {1:.0f}) is incomplete.".format(tile_rindex, tile_cindex))
    if selection == "Martinis":
        return (average, stdev), hand
    elif selection == "Chini":
        return (a, bc, sr), hand
    else:
        return None, hand  # TODO: modify for other selection methods


def tiled_thresholding(image, selection='Martinis', t_method=['KI', 'Otsu'], tile_dim=[200, 200], n_final=5,
                       hand_matrix=None, hand_t=100, directory_figures=None, accuracy=200, incomplete_tile_warning=True):
    """ Apply tiled thresholding
    
    Inputs:
    image: nd array
        Array of pixel values.
    selection: str (default='Martinis')
        Method for tile selection. Currently only option is 'Martinis'.
    t_method: list (default=['KI', 'Otsu'])
        List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    n_final: int (default=5)
        Number of tiles to select.
    hand_matrix: ndarray or None (default=None)
        Array of HAND values.
    hand_t: float (default=100)
        Maximum HAND value allowed for threshold selection.
    directory_figures: str or None (default=None)
        If not None, figure is saved to specified directory.
    incomplete_tile_warning: bool (default=True)
        Whether to give a warning when incomplete tiles are encountered.
    Outputs:
    tile_ki: nd array
        Array of KI thresholds.
    tile_o: nd array
        Array of Otsu thresholds.
    average: nd array
        Array of tile averages.
    stdev: nd array
        Array of tiled st. devs.
    hand: nd array
        Array of tile mean HAND values.
    """
    tile_dim = np.array(tile_dim)
    if selection == "Martinis":
        # Tile properties and selection
        (average, stdev), hand = tile_vars(image, selection=selection, t_method=t_method, tile_dim=tile_dim,
                                           accuracy=accuracy, hand_matrix=hand_matrix,
                                           directory_figures=directory_figures,
                                           incomplete_tile_warning=incomplete_tile_warning)
        q = np.nanpercentile(stdev, 95)
        stdev[average > np.nanmean(average)] = np.nan
        if hand_matrix:
            stdev[hand > hand_t] = np.nan
        i_r, i_c = np.where(stdev > q)  # select tiles with stdev > 95-percentile
        while len(i_r) == 0:
            tile_dim = tile_dim//2
            print('Tile dimensions halved.')
            (average, stdev), hand = tile_vars(image, selection=selection, t_method=t_method, tile_dim=tile_dim,
                                               accuracy=accuracy, hand_matrix=hand_matrix,
                                               directory_figures=directory_figures,
                                               incomplete_tile_warning=incomplete_tile_warning)
            q = np.percentile(stdev, 95)
            i_r, i_c = np.where(stdev > q)
        sorted_indices = np.argsort(stdev[i_r, i_c])[::-1]
        i_r = i_r[sorted_indices]
        i_c = i_c[sorted_indices]
        # Tile threshold values
        tile_ki = []
        tile_o = []
        for tile_rindex, tile_cindex in zip(i_r, i_c):
            tile = image[tile_rindex * tile_dim[0]:min((tile_rindex + 1) * tile_dim[0], image.shape[0]),
                         tile_cindex * tile_dim[1]:min((tile_cindex + 1) * tile_dim[1], image.shape[1])]
            if directory_figures:
                fig, ax = plt.subplots()
                ax.imshow(tile, cmap="gray", vmin=-20, vmax=0)
                plt.savefig(os.path.join(directory_figures, "Tile{}-{}_image.png".format(tile_rindex, tile_cindex)))
                plt.close(fig)
            if "KI" in t_method:
                if directory_figures:
                    fig_filename = os.path.join(directory_figures, "Tile{}-{}_KI.png".format(tile_rindex, tile_cindex))
                else:
                    fig_filename = None
                tile_ki.append(apply_ki(tile, accuracy=accuracy, t_min=-30, t_max=-5, fig_filename=fig_filename))
            if "Otsu" in t_method:
                if directory_figures:
                    fig_filename = os.path.join(directory_figures, "Tile{}-{}_Otsu.png".format(tile_rindex, tile_cindex))
                else:
                    fig_filename = None
                tile_o.append(apply_otsu(tile, accuracy=accuracy, fig_filename=fig_filename))
        tile_ki = np.array(tile_ki)
        tile_o = np.array(tile_o)
        i_r = i_r[~np.isnan(tile_ki)]
        i_c = i_c[~np.isnan(tile_ki)]
        tile_o = tile_o[~np.isnan(tile_ki)]
        tile_ki = tile_ki[~np.isnan(tile_ki)]
        if i_r.size > n_final:
            tile_selection = [i_r[:n_final], i_c[:n_final]]
            tile_ki = tile_ki[:n_final]
            tile_o = tile_o[:n_final]
        else:
            tile_selection = [i_r, i_c]
        # Global threshold and quality indicator
        if 'KI' in t_method:
            t_ki = np.mean(tile_ki)
            s = np.std(tile_ki)
            if s > 5:
                print('Histogram merge necessary for KI.')
                pixel_selection = np.empty(0)
                for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]):
                    tile = image[tile_rindex*tile_dim[0]:min((tile_rindex+1)*tile_dim[0], image.shape[0]),
                                 tile_cindex*tile_dim[1]:min((tile_cindex+1)*tile_dim[1], image.shape[1])]
                    pixel_selection = np.append(pixel_selection, tile.ravel())
                    del tile
                t_ki = apply_ki(pixel_selection)
        if 'Otsu' in t_method:
            t_otsu = np.mean(tile_o)
            s = np.std(tile_o)
            if s > 5:
                print('Histogram merge necessary for Otsu.')
                pixel_selection = np.empty(0)
                for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]):
                    tile = image[tile_rindex*tile_dim[0]:min((tile_rindex+1)*tile_dim[0], image.shape[0]),
                                 tile_cindex*tile_dim[1]:min((tile_cindex+1)*tile_dim[1], image.shape[1])]
                    pixel_selection = np.append(pixel_selection, tile.ravel())
                    del tile
                t_otsu = apply_otsu(pixel_selection)
    elif selection == "Chini":
        # Tile properties and selection
        (a, bc, sr), hand = tile_vars(image, selection=selection, t_method=t_method,
                                                         tile_dim=tile_dim, accuracy=accuracy,
                                                         hand_matrix=hand_matrix, directory_figures=directory_figures,
                                                         incomplete_tile_warning=incomplete_tile_warning)
        tile_selection = np.array(np.where((a > 2) & (bc > 0.99) & (sr > 0.1)))
        # Global threshold value
        pixel_selection = np.empty(0)
        for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]):
            tile = image[tile_rindex * tile_dim[0]:min((tile_rindex + 1) * tile_dim[0], image.shape[0]),
                         tile_cindex * tile_dim[1]:min((tile_cindex + 1) * tile_dim[1], image.shape[1])]
            pixel_selection = np.append(pixel_selection, tile.ravel())
            del tile
        if 'KI' in t_method:
            t_ki = apply_ki(pixel_selection)
        if "Otsu" in t_method:
            t_otsu = apply_otsu(pixel_selection, accuracy=accuracy)
    # if directory_figures:
    #     _ = show_tileselection(image, tile_selection)
    #     plt.savefig(os.path.join(directory_figures, "Thresholding_TileSelection.png"))
    #     plt.close(fig)
    # Return
    if 'KI' in t_method and 'Otsu' in t_method:
        return [t_ki, t_otsu], tile_selection
    elif 'KI' in t_method:
        return t_ki, tile_selection
    elif 'O' in t_method:
        return t_otsu, tile_selection


def show_tileselection(image, tile_selection, tile_dim=[200, 200]):
    """Plot image tiles and indicate selection
    
    Inputs:
    image: nd array
        Array of pixel values.
    tile_selection: list
        List of row and column indices selected tiles.
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    Outputs:
    fig: handle to figure
    ax: handle to axis
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for r in np.arange(image.shape[0]+1, step=200):
        ax.plot([0, image.shape[1]], [r, r], 'r')
    for c in np.arange(image.shape[1]+1, step=200):
        ax.plot([c, c], [0, image.shape[0]], 'r')    
    for tiler, tilec in zip(tile_selection[0], tile_selection[1]):
        ax.plot([tilec*tile_dim[0], tilec*tile_dim[0]], [tiler*tile_dim[0], (tiler+1)*tile_dim[0]], color=[0, 1, 0])
        ax.plot([(tilec+1)*tile_dim[0], (tilec+1)*tile_dim[0]], [tiler*tile_dim[0], (tiler+1)*tile_dim[0]],
                color=[0, 1, 0])
        ax.plot([tilec*tile_dim[0], (tilec+1)*tile_dim[0]], [tiler*tile_dim[0], tiler*tile_dim[0]], color=[0, 1, 0])
        ax.plot([tilec*tile_dim[0], (tilec+1)*tile_dim[0]], [(tiler+1)*tile_dim[0], (tiler+1)*tile_dim[0]],
                color=[0, 1, 0])
    ax.set_xlim(-5, image.shape[1]+5)
    ax.set_ylim(image.shape[0]+5, -5)
    ax.axis('off')
    return fig, ax


def show_thresholds(image, t, t_labels=("KI", "Otsu")):
    """Plot image histogram and thresholds
    
    Inputs:
    image: nd array
        Array of pixel values.
    t: list
        List of threshold values.
    tile_labels: tuple
        Tuple of threshold labels for legend.
    Outputs:
    fig: handle to figure
    ax: handle to axis
    """
    accuracy = 200
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    fig, ax = plt.subplots()
    ax.bar(g, h, width=bin_width, color=[0.5, 0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
    ps = []
    for t_i, t_value in enumerate(t):
        if "KI" in t_labels[t_i]:
            color = [1, 0.5, 0]
        elif "Otsu" in t_labels[t_i]:
            color = [0, 0.7, 0]
        if "tiled" in t_labels[t_i]:
            linestyle = "dashed"
        else:
            linestyle = "solid"
        p = ax.plot([t_value, t_value], [0, np.nanmax(h)], color=color, linestyle=linestyle)
        ps.append(p[0])
    _ = fig.legend(tuple(ps), t_labels, 'upper right', framealpha=1)
    return fig, ax


def get_ashman(mu1, sigma1, mu2, sigma2):
    return np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(sigma1**2 + sigma2**2)


def get_bc(x, y, p):
    yf = two_gaussians(x, *p)
    return np.sum(np.sqrt(y) * np.sqrt(yf))


def get_sr(a1, sigma1, a2, sigma2):
    return min(a1*sigma1*np.sqrt(2*np.pi), a2*sigma2*np.sqrt(2*np.pi)) / \
           max(a1*sigma1*np.sqrt(2*np.pi), a2*sigma2*np.sqrt(2*np.pi))


def one_gaussion(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2))


def get_hist(image, accuracy=200):
    """Get density histogram of image"""
    image = image.ravel()
    y, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    x = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    return x, y


def fit_two_gaussians(x, y, t_otsu, fig_filename=None):
    """Fit two Gaussians to x, y"""
    p0 = []
    for subselection in np.arange(2):
        if subselection == 0:
            subselection_x = x[x < t_otsu]
        else:
            subselection_x = x[x > t_otsu]
        mu0 = np.mean(subselection_x)
        sigma0 = np.std(subselection_x)
        a0 = y[abs(x - mu0) == np.min(abs(x - mu0))][0]
        p0.extend([a0, mu0, sigma0])
    popt, pcov = curve_fit(two_gaussians, x, y, p0=p0)
    if fig_filename:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot([t_otsu, t_otsu], [0, 1.05*np.max(y)], 'k')
        ax.plot(x, two_gaussians(x, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5]), 'g')
        ax.plot(x, two_gaussians(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), 'r')
        ax.plot(x, one_gaussion(x, popt[0], popt[1], popt[2]), '--r')
        ax.plot(x, one_gaussion(x, popt[3], popt[4], popt[5]), '--r')
        plt.savefig(fig_filename)
        plt.close(fig)
    return popt


def find_valley(x, y):
    """ Find local minimum values of x based on y"""
    valleys = []
    x_local_max = x[argrelextrema(y, np.greater)]
    x_local_min = x[argrelextrema(y, np.less)]
    for m in x_local_min:
        if (np.sum(x_local_max < m) >= 1) & (np.sum(x_local_max > m) >= 1):
            valleys.append(m)
    return valleys
