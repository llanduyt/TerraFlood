# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Main and ancillary functions of the TerraFlood algorithm
"""

import os
import numpy as np
from datetime import datetime as dt
from scipy import ndimage as nd
from scipy import stats
import contextily as cx
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from sklearn import preprocessing
from skimage import measure
from skimage import morphology as morph
from skimage import segmentation as seg

wgs = "epsg:4326"
utm = "epsg:32631"
s1_resolution = 10

nd_pres = 99
nd_map = -1

l_dl = 0
l_pw = 1
l_of = 2
l_pf = 3
l_ltf = 4
l_fv = 5
l_ff = 6
l_if = 7

dir_root = "/data/users/Private/flood_vmm/"
dir_lc = os.path.join(dir_root, "Ancillary_Data", "BBK5_15", "GeoTIFF")
dir_dem = os.path.join(dir_root, "Ancillary_Data", "DHM5", "GEOTIFF")


def make_floodmap(floodmap, extent, filename, flood_location, flood_date, basemap_source=cx.providers.Stamen.TonerLite):
    """
    Make plot of flood map and save to .png figure

    Inputs:
    floodmap: nd array
        Array of flood classes
    extent: array-like
        x_min, y_min, x_max, y_max of the area to plot
    filename: string
        Filename to which to save the figure
    flood location: string
        Location of flood map, used in figure title
    flood_date: datetime object
        Datetime of flood map, used in figure title
    basemap_source: contextily provider
        Contextily basemap provider
    """
    colorlist = ['black', 'coral', 'skyblue', 'blue', 'royalblue', 'navy', 'springgreen', 'green', 'grey']
    cmap_cc = colors.ListedColormap(colorlist)
    leg_labels = ["PW", "OF", "PF", "LTF", "FV", "FF", "IF"]
    leg_handles = []
    for color in colorlist[2:]:
        leg_handles.append(Line2D([0], [0], color=color, lw=5))

    map_extent = (extent[0], extent[2], extent[1], extent[3])
    floodmap = np.ma.masked_where(np.isin(floodmap, [l_dl]), floodmap)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(floodmap, extent=map_extent, cmap=cmap_cc, vmin=-1, vmax=7)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    cx.add_basemap(ax, crs=utm, source=basemap_source)
    ax.imshow(floodmap, extent=map_extent, cmap=cmap_cc, vmin=-1, vmax=7)
    ax.legend(leg_handles, leg_labels, bbox_to_anchor=(0.5, -0.05), loc=9, ncol=7, fontsize=14)
    plt.text(0.5, 1.03, "Flooding in {} on {}".format(flood_location, dt.strftime(flood_date, "%Y-%m-%d %H:%M")),
             fontsize=16, horizontalalignment="center", transform=ax.transAxes)
    plt.savefig(filename)
    plt.close(fig)


def make_map_sar(array_s1, extent, filename, flood_location, flood_date, sar_min=-20, sar_max=0):
    """
    Make plot of SAR composite and save to .png figure

    Inputs:
    array_s1: nd array
        SAR array. First dimension should equal 4 (VH_ref, VV_ref, VH_flood, VV_flood). B1 and 3 are used for composite.
    extent: array-like
        x_min, y_min, x_max, y_max of the area to plot
    filename: string
        Filename to which to save the figure
    flood location: string
        Location of flood map, used in figure title
    flood_date: datetime object
        Datetime of flood map, used in figure title
    sar_min & sar_max: float
        Lower and upper color limit for visualisation
    """
    s1 = np.array([array_s1[1], array_s1[3], array_s1[3]])
    s1[s1 < sar_min] = sar_min
    s1[s1 > sar_max] = sar_max
    s1 = (s1 - sar_min) / (sar_max - sar_min)
    s1[:, np.isnan(s1[0])] = 0
    s1[:, np.isnan(s1[2])] = 0
    s1 = np.swapaxes(np.swapaxes(s1, 0, 1), 1, 2)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(s1.astype("float"), extent=(extent[0], extent[2], extent[1], extent[3]), vmin=0, vmax=1)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.legend([], [], bbox_to_anchor=(0.5, -0.05), loc=9, ncol=6, fontsize=14)
    plt.text(0.5, 1.03, "SAR image {} on {}".format(flood_location, dt.strftime(flood_date, "%d-%m-%Y %H:%M")),
             fontsize=16, horizontalalignment="center", transform=ax.transAxes)
    plt.savefig(filename)
    plt.close(fig)


def apply_mmu(array, mmu, background=0, connectivity=2):
    """ 
    Remove objects with a pixel size smaller than mmu from map
    
    Inputs
    array: nd array
        array of discrete values
    mmu: int
        minimal area (in pixels) for objects to be maintained
    background: int
        background value in array, not considered for object labeling
    connectivity: int
        connectivity used for object labeling (see skimage.measure.label)
    Outputs
    array: nd array
        updated array
    """
    ids_array = measure.label(array, background=background, connectivity=connectivity)
    props = measure.regionprops(ids_array)
    ids_list = np.array([el["label"] for el in props])
    sizes = np.array([el["area"] for el in props])
    array[np.isin(ids_array, ids_list[sizes < mmu])] = background
    return array


def combine_footprints(array1, array2, mmu_core=1):
    """ 
    Combine arrays and only maintain objects with a union >= mmu_core

    Inputs
    array1: nd array
        array of discrete values
    array2: nd array
        array of discrete values
    mmu_core: int
        minimal area (in pixels) for common part of objects to be maintained
    Outputs
    combo: nd array
        combined array
    """
    core = apply_mmu(np.min(np.array([array1, array2]), axis=0), mmu_core)
    ids1 = measure.label(array1, background=0, connectivity=2)
    ids1_core = list(np.unique(ids1[core > 0]))
    ids2 = measure.label(array2, background=0, connectivity=2)
    ids2_core = list(np.unique(ids2[core > 0]))
    combo = np.max(np.array([np.isin(ids1, ids1_core), np.isin(ids2, ids2_core)]), axis=0)
    return combo


def rg(floodmap, seed_labels, target_label, grow_criterium, selem=morph.selem.disk(1)):
    """
    Apply region growing
    
    Inputs:
    floodmap: nd array
        array of discrete flood classes
    seed_labels: list
        list of flood class values to consider as seeds
    target_label: int
        flood class value given to grown pixels
    grow_criterium: nd array, bool
        array indicating which pixels can be considered for RG
    selem: nd array, bool
         moving window indicating which pixels to select as neighbour seeds
    Outputs:
    floodmap: nd array
        updated flood map
    """
    num_changes = 10
    num_changes_total = 0
    its = 0
    while num_changes > 0:
        its += 1
        floodmap_old = np.copy(floodmap)
        neighbours = morph.binary_dilation(np.isin(floodmap, seed_labels), selem=selem) & (floodmap == 0)
        floodmap[neighbours & grow_criterium] = target_label
        num_changes = np.sum(floodmap != floodmap_old)
        num_changes_total += num_changes
    # print("{} - Changed state of {} pixels over {} iterations".format(dt.now(), num_changes_total, its))
    return floodmap


def rg_ff(floodmap, trees, ogp, ogf, odp, odf, seed_labels, ff_inclusion="risk-depth", dem=None):
    """
    Apply region growing to indicate probably flooded forests
    
    Inputs
    floodmap: nd array
        array of discrete flood classes
    trees: nd array, bool
        array indicating tree presence
    ogp: nd array
        array indicating pluvial flood susceptibility classes
    ogf: nd array
        array indicating fluvial flood susceptibility classes
    odp: nd array
        array indicating pluvial flood depth classes
    odf: nd array
        array indicating fluvial flood depth classes
    seed_labels: list
        list of flood class values to consider as seeds
    ff_inclusion: string
        string indicating which forest pixels to include; should equal "all", "risk", "depth" or "risk-depth" (default)
    dem: nd array
        array of elevation values
    Outputs:
    floodmap: nd array
        updated flood map
    """
    if ff_inclusion == "all":
        del odp, odf
        og = np.max(np.array([ogp, ogf]), axis=0)
        del ogp, ogf
        trees_with_risk = morph.binary_dilation(trees == 1, selem=morph.selem.rectangle(3, 3)) & (og > 0)
        trees_label = morph.label(trees_with_risk, connectivity=1)
        neighbouring_trees_with_risk = trees_label[
            trees_with_risk &
            morph.binary_dilation(np.isin(floodmap, seed_labels), selem=morph.selem.disk(1)) &
            (floodmap == 0)]
        floodmap[np.isin(trees_label, neighbouring_trees_with_risk) & (floodmap == 0)] = l_ff
    elif ff_inclusion == "risk":
        seed_objects = measure.label(np.isin(floodmap, seed_labels), background=0)  # label seed regions
        s_objects_label = np.unique(seed_objects[seed_objects != 0])  # get seed labels
        s_objects_ogp = np.array([np.argmax(np.bincount(ogp[seed_objects == label])) for label in s_objects_label])  # get main ogp level per region
        s_objects_ogf = np.array([np.argmax(np.bincount(ogf[seed_objects == label])) for label in s_objects_label])  # get main ogf level per region
        for og, s_objects_og in zip([ogp, ogf], [s_objects_ogp, s_objects_ogf]):
            for risk in [1, 2, 3]:
                trees_with_risk = morph.binary_dilation(trees == 1, selem=morph.selem.rectangle(3, 3)) & (og == risk)  # get trees with specified risk level
                trees_label = morph.label(trees_with_risk, connectivity=1)  # label tree regions with risk
                neighbouring_trees_with_risk = trees_label[  # get label of tree regions with risk touches seed region with specified risk level
                    trees_with_risk &
                    morph.binary_dilation(np.isin(seed_objects, s_objects_label[s_objects_og == risk]),
                                          selem=morph.selem.disk(1)) &
                    (floodmap == 0)]
                floodmap[np.isin(trees_label, neighbouring_trees_with_risk) & (floodmap == 0)] = l_ff
    elif ff_inclusion == "depth":
        nd_val = 9999
        if dem is None:
            print("Error! To consider flood depth, DEM should be provided!")
        od = np.max(np.array([odp, odf]), axis=0)
        del odp, odf
        wl = nd.median_filter(od + dem, footprint=morph.selem.disk(1), mode="nearest")  # water level height = elevation + flood depth
        wl[od == 0] = nd_val  # make sure areas with no flood depth are not considered for RG
        trees_buffered = morph.binary_dilation(trees == 1, selem=morph.selem.rectangle(3, 3))
        num_changes = 10
        its = 0
        seed_labels += [l_ff]
        while num_changes > 0:
            its += 1
            floodmap_old = np.copy(floodmap)
            seeds = np.isin(floodmap, seed_labels)
            wl_seeds = seeds * wl
            wl_seeds[wl_seeds == 0] = nd_val
            neighbours_seed_wl = nd.minimum_filter(wl_seeds, footprint=morph.selem.disk(1), mode="constant", cval=0)
            neighbours_seed_wl[neighbours_seed_wl != nd_val] = neighbours_seed_wl[neighbours_seed_wl != nd_val] - 0.1  # subtract 0.1 to account for uncertainty due to discrete classes
            neighbours = morph.binary_dilation(seeds, selem=morph.selem.disk(1)) & (seeds == 0)
            floodmap[neighbours & trees_buffered & (wl != nd_val) & (wl >= neighbours_seed_wl)] = l_ff
            num_changes = np.sum(floodmap != floodmap_old)
    elif ff_inclusion == "risk-depth":
        nd_val = 9999
        if dem is None:
            print("Error! To consider flood depth, DEM should be provided!")
        od = np.max(np.array([odp, odf]), axis=0)
        del odp, odf
        wl = nd.median_filter(od + dem, footprint=morph.selem.disk(1), mode="nearest")  # water level height = elevation + flood depth
        wl[od == 0] = nd_val  # make sure areas with no flood depth are not considered for RG
        trees_buffered = morph.binary_dilation(trees == 1, selem=morph.selem.rectangle(3, 3))
        num_changes = 10
        its = 0
        seed_labels += [l_ff]
        seed_objects = measure.label(np.isin(floodmap, seed_labels), background=0)  # label seed regions
        seed_objects_og = np.zeros(seed_objects.shape)
        for label in np.unique(seed_objects[seed_objects != 0]):
            seed_objects_og[seed_objects == label] = np.max([np.argmax(np.bincount(ogp[seed_objects == label])), np.argmax(np.bincount(ogf[seed_objects == label]))])  # max of main ogp/ogf level per region
        seed_objects_og[seed_objects_og == 0] = nd_val
        og = np.max(np.array([ogp, ogf]), axis=0)
        og[og == 0] = nd_val
        del ogp, ogf, seed_objects
        while num_changes > 0:
            its += 1
            floodmap_old = np.copy(floodmap)
            seeds = np.isin(floodmap, seed_labels)
            prop_seeds = seeds * wl
            prop_seeds[prop_seeds == 0] = nd_val
            neighbours_seed_wl = nd.minimum_filter(prop_seeds, footprint=morph.selem.disk(1), mode="constant",
                                                   cval=nd_val)
            neighbours_seed_wl[neighbours_seed_wl != nd_val] = neighbours_seed_wl[neighbours_seed_wl != nd_val] - 0.1  # subtract 0.1 to account for uncertainty due to discrete classes
            neighbours_seed_og = nd.minimum_filter(seed_objects_og, footprint=morph.selem.disk(1), mode="constant",
                                                   cval=nd_val)
            neighbours = morph.binary_dilation(seeds, selem=morph.selem.disk(1)) & (seeds == 0)
            floodmap[neighbours & trees_buffered & (wl != nd_val) & (wl >= neighbours_seed_wl) & (og != nd_val) & (og >= neighbours_seed_og)] = l_ff
            seed_objects_og[(seed_objects_og == nd_val) & (floodmap == l_ff)] = neighbours_seed_og[(seed_objects_og == nd_val) & (floodmap == l_ff)]
            num_changes = np.sum(floodmap != floodmap_old)
    # print("{} - Changed state of {} pixels".format(dt.now(), np.sum(floodmap == l_ff)))
    return floodmap


def remove_smallobjects(floodmap, class_labels, mmu=5):
    """
    Reclassify small objects
    
    Inputs
    floodmap: nd array
        array of discrete flood classes
    class_labels: list
        flood classes to consider for small object reclassification
    mmu: int
        objects equal to or smaller than this value are removed
    Outputs
    floodmap: nd array
        array of updated flood classes
    """
    def find_mode(labeled_pixels):
        """Get most occurring value"""
        return stats.mode(labeled_pixels, axis=None)[0][0]

    def modal(array):
        """Get most occurring value excluding DL and IF"""
        array = [el for el in array if el not in [l_dl, l_if]]
        if len(array) > 0:
            return stats.mode(array, axis=None)[0][0]
        else:
            return l_dl

    px_modn = nd.generic_filter(floodmap, modal, footprint=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                mode="constant", cval=l_dl)
    for l_class in class_labels:
        class_obs, n_labels = nd.label(floodmap == l_class, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        class_ob_labels, class_objects_sizes = np.unique(class_obs[class_obs != 0], return_counts=True)
        class_ob_labels = class_ob_labels[class_objects_sizes <= mmu]
        if len(class_ob_labels) > 0:
            class_ob_modn = nd.labeled_comprehension(px_modn, class_obs, class_ob_labels, find_mode, int, 0)
            for l_ob, mod_ob in zip(class_ob_labels, class_ob_modn):
                floodmap[class_obs == l_ob] = mod_ob
    return floodmap


def fill_holes(floodmap, flood_vv, flood_vh, ref_vv, ref_vh, t_vv, t_vh, class_labels_holes, mmu=5):
    """
    Reclassify holes (i.e. small non-flood objects)
    
    Inputs
    floodmap: nd array
        array of discrete flood classes
    flood_vv: nd array
        array of flood VV backscatter values
    flood_vh: nd array
        array of flood VH backscatter values
    ref_vv: nd array
        array of ref. VV backscatter values
    ref_vh: nd array
        array of ref. VH backscatter values
    t_vv: float
        VV threshold value
    t_vh: float
        VH threshold value
    class_labels_holes: list
        class values to consider for hole filling
    mmu: int 
        objects equal to or smaller than this value are removed
    Outputs
    floodmap: nd array
        array of updated flood classes
    """
    num_changes = 10
    while num_changes > 0:
        floodmap_old = np.copy(floodmap)
        holes = morph.label(np.isin(floodmap, class_labels_holes), connectivity=1) 
        h_area = np.array([el["area"] for el in measure.regionprops(holes)])
        h_label = np.array([el["label"] for el in measure.regionprops(holes)])
        h_fvv = np.array([el["mean_intensity"] for el in measure.regionprops(holes, flood_vv)])
        h_fvh = np.array([el["mean_intensity"] for el in measure.regionprops(holes, flood_vh)])
        h_rvv = np.array([el["mean_intensity"] for el in measure.regionprops(holes, ref_vv)])
        h_rvh = np.array([el["mean_intensity"] for el in measure.regionprops(holes, ref_vh)])
        h_tofill = (h_area <= mmu) | ((h_fvv < t_vv) & (h_fvh < t_vh+1)) | \
                       ((h_fvh < t_vh) & (h_fvv < t_vv+1))
        h_label = h_label[h_tofill]
        h_area = h_area[h_tofill]
        h_fvv = h_fvv[h_tofill]
        h_fvh = h_fvh[h_tofill]
        for i_l, l in enumerate(h_label):
            h_neighclasses = floodmap[morph.binary_dilation(holes == l, selem=morph.selem.rectangle(3, 3)) ^ (holes == l)]
            if (((h_fvv[i_l] < t_vv) & (h_fvh[i_l] < t_vh+1)) | ((h_fvh[i_l] < t_vh) & (h_fvv[i_l] < t_vv+1))) & \
               (((h_rvv[i_l] < t_vv) & (h_rvh[i_l] < t_vh+1)) | ((h_rvh[i_l] < t_vh) & (h_rvv[i_l] < t_vv+1))):
                if len(h_neighclasses) > 0:
                    if np.bincount(h_neighclasses).argmax() in [l_pw, l_of]:
                        floodmap[holes == l] = np.bincount(h_neighclasses).argmax()
                elif l_pw in h_neighclasses:
                    floodmap[holes == l] = l_pw
                else:
                    floodmap[holes == l] = l_pf
            elif ((h_fvv[i_l] < t_vv) & (h_fvh[i_l] < t_vh+1)) | ((h_fvh[i_l] < t_vh) & (h_fvv[i_l] < t_vv+1)):
                if len(h_neighclasses) > 0:
                    if np.bincount(h_neighclasses).argmax() in [l_pw, l_of]:
                        floodmap[holes == l] = np.bincount(h_neighclasses).argmax()
                elif l_of in h_neighclasses:
                    floodmap[holes == l] = l_of
                else:
                    floodmap[holes == l] = l_pf
            elif h_area[i_l] <= mmu:
                h_neighclasses = [el for el in h_neighclasses if el != l_ff]
                if len(h_neighclasses) > 0:
                    floodmap[holes == l] = np.bincount(h_neighclasses).argmax()
        num_changes = np.sum(floodmap != floodmap_old)
    return floodmap


def apply_quickseg(image, ratio=1.0, maxdist=4, kernel_window_size=7):
    """
    Apply quickshift segmentation for the specified set of parameters.

    Inputs:
    image: nd array
        Input array for segmentation. Dimensions should be rows x columns x bands.
    image_bandnames: list
        List specifying the band order in image. Possible elements are:
        rVH, rVV, fVH, fVV, rR, fR, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    image_metadata: dict
        Dictionary specifying image metdata, output of rasterio meta property.
    ratio: float (default=1.0)
        Ratio balancing color-space proximity and image-space proximity, should be between 0 and 1. Higher values give more weight to color-space.
    maxdist: float (default=4)
        Cut-off point for data distances. Higher means fewer clusters.
    kernel_window_size: int (default=7)
        Size of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters. Minimum equals 7x7.
    directory_output: str or None (default=None)
        If not None, output will be saved to specified path.
    Outputs:
    segments_quick: geopandas GeoDataFrame
        Dataframe of segments, columns = ["geometry", "DN"]
    image_segmented: nd array
        array of labeled segments
    """
    # Check image dimensions
    no_rows, no_cols, no_bands = image.shape
    if no_bands > no_rows:
        print("Warning! Image dimensions should be row x column x bands. Current dimensions are {}x{}x{}, "
              "which seems wrong. Swapping axes...".format(no_rows, no_cols, no_bands))
        image = np.transpose(image, (1, 2, 0))
        no_rows, no_cols, no_bands = image.shape
    # Normalize data
    for band_index in np.arange(no_bands):
        if np.nanstd(image[:, :, band_index]) != 1 or np.nanmean(image[:, :, band_index]) != 0:
            band = image[:, :, band_index]
            band[np.isfinite(band)] = preprocessing.StandardScaler().fit_transform(
                band[np.isfinite(band)].reshape(-1, 1))[:, 0]
            image[:, :, band_index] = band
    # Segmentation
    kernel_size = (kernel_window_size - 1) / 6
    image_segmented = seg.quickshift(image.astype('double'), ratio=ratio, max_dist=maxdist, kernel_size=kernel_size,
                                     convert2lab=False)
    image_segmented += 1  # add 1 to avoid background value 0
    image_segmented = measure.label(image_segmented, connectivity=1)
    return image_segmented


def apply_terraflood(flood_vv, flood_vh, ref_vv, ref_vh, inc_vv, inc_r, trees, pw, ogp, ogf, odp, odf, dem, t_vv, t_vh,
                     t_incvv=3, t_incr=3, ratio=1.0, kernel_ws=7, maxdist=3, mmu=10, mmu_holes=5, pf_selection="all",
                     ff_inclusion="all", pw_selection="all", pw_perc=0.5, fv_seeds="all"):
    """
    Apply object-based version of TerraFlood algorithm
    
    Inputs
    flood_vv: nd array
        array of flood VV backscatter values
    flood_vh: nd array
        array of flood VH backscatter values
    ref_vv: nd array
        array of ref. VV backscatter values
    ref_vh: nd array
        array of ref. VH backscatter values
    inc_vv: nd array
        array of increase in VV backscatter values
    inc_r: nd array
        array of increase in (VV-VH) backscatter values
    trees: nd array, bool
        array indicating tree presence
    pw: nd array, bool
        array indicating permanent water presence
    ogp: nd array
        array indicating pluvial flood susceptibility classes
    ogf: nd array
        array indicating fluvial flood susceptibility classes
    odp: nd array
        array indicating pluvial flood depth classes
    odf: nd array
        array indicating fluvial flood depth classes
    dem: nd array
        array of elevation values
    t_vv: float
        VV threshold value
    t_vh: float
        VH threshold value
    t_incvv: float
        incVV threshold value
    t_incr: float
        incR threshold value
    ratio: float (default=1.0)
        ratio for apply_quickseg
    kernel_window_size: int (default=7)
        size of Gaussian kernel for apply_quickseg
    maxdist: float (default=4)
        cut-off point for apply_quickseg
    mmu: int
        minimal mapping unit for core flooding
    mmu_holes: int
        minimal mappint unit for holes
    pf_selection: string
        how to filter the PF class, one of ["all", "risk", "none"]
    ff_inclusion: string
        how to include the FF class, one of ["all", "risk", "depth", "risk-depth"]
    pw_selection: string
        how to filter the PW class, one of ["all", "bbk", "bbk-risk"]
    pw_perc: int
        percentage value [0-100] used for PW/PF filtering
    fv_seeds: list
        list of flood classes to consider as seeds for FV region growing
    Outputs:
    floodmap: nd array
        TerraFlood flood map (classes defined on L35-42)
    """
    # Segmentation
    seg_labels = apply_quickseg(np.transpose(np.array([ref_vh, ref_vv, flood_vh, flood_vv]), (1, 2, 0)).copy(),
                                    ratio=ratio, maxdist=maxdist, kernel_window_size=kernel_ws)

    # Removal single-pixel objects
    labels, sizes = np.unique(seg_labels, return_counts=True)
    nr, nc = seg_labels.shape
    singular_labels = labels[sizes == 1]
    diff_hor = np.sum(abs(np.diff(np.array([ref_vh, ref_vv, flood_vh, flood_vv]), axis=1)), axis=0)
    diff_ver = np.sum(abs(np.diff(np.array([ref_vh, ref_vv, flood_vh, flood_vv]), axis=2)), axis=0)
    diff_all = np.array([np.concatenate((np.ones((1, nc)) * 99, diff_hor), axis=0),
                         np.concatenate((diff_hor, np.ones((1, nc)) * 99), axis=0),
                         np.concatenate((np.ones((nr, 1)) * 99, diff_ver), axis=1),
                         np.concatenate((diff_ver, np.ones((nr, 1)) * 99), axis=1)])
    most_similar = np.argmin(diff_all, axis=0)
    ind_neigh = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    while len(singular_labels) > 0:
        singular_r, singular_c = np.where(np.isin(seg_labels, singular_labels))
        for r, c in zip(singular_r, singular_c):
            seg_labels[r, c] = seg_labels[r + ind_neigh[most_similar[r, c], 0], c + ind_neigh[most_similar[r, c], 1]]
        labels, sizes = np.unique(seg_labels, return_counts=True)
        singular_labels = labels[sizes == 1]
    del diff_hor, diff_ver, diff_all, most_similar, ind_neigh, singular_labels, singular_r, singular_c

    # Extract features
    seg_labels = measure.label(seg_labels, connectivity=1) - 1
    labels = np.unique(seg_labels)
    ref_vh_ob_list = nd.labeled_comprehension(ref_vh, seg_labels, labels, np.mean, float, -99)
    ref_vh_ob = ref_vh_ob_list[seg_labels]
    ref_vv_ob_list = nd.labeled_comprehension(ref_vv, seg_labels, labels, np.mean, float, -99)
    ref_vv_ob = ref_vv_ob_list[seg_labels]
    flood_vh_ob_list = nd.labeled_comprehension(flood_vh, seg_labels, labels, np.mean, float, -99)
    flood_vh_ob = flood_vh_ob_list[seg_labels]
    flood_vv_ob_list = nd.labeled_comprehension(flood_vv, seg_labels, labels, np.mean, float, -99)
    flood_vv_ob = flood_vv_ob_list[seg_labels]
    del seg_labels, ref_vh_ob_list, ref_vv_ob_list, flood_vh_ob_list, flood_vv_ob_list

    # Thresholding
    # print("{} - Pixel-based thresholding...".format(dt.now()))
    pw_px = (flood_vv < t_vv) & (flood_vh < t_vh) & (ref_vv < t_vv) & (ref_vh < t_vh)
    flood_px = (flood_vv < t_vv) & (flood_vh < t_vh) & (pw_px == 0)
    # print("{} - Object-based thresholding...".format(dt.now()))
    pw_ob = (flood_vv_ob < t_vv) & (flood_vh_ob < t_vh) & (ref_vv_ob < t_vv) & (ref_vh_ob < t_vh)
    flood_ob = (flood_vv_ob < t_vv) & (flood_vh_ob < t_vh) & (pw_ob == 0)
    # print("{} - Combination of pixel- and object-based thresholding...".format(dt.now()))
    pw_combo = combine_footprints(pw_px, pw_ob)
    flood_combo = combine_footprints(flood_px, flood_ob)
    floodmap = np.zeros(pw_combo.shape, dtype="int")
    floodmap[flood_combo] = l_of
    floodmap[pw_combo] = l_pw
    del pw_combo, flood_combo

    # Region grow into groups of pixels satisfying class definitions
    # print("{} - Region growing into PW areas...".format(dt.now()))
    floodmap = rg(floodmap, [l_pw, l_of], l_pw, (pw_px & pw_ob))
    # print("{} - Region growing into OF areas...".format(dt.now()))
    floodmap = rg(floodmap, [l_pw, l_of], l_of, (flood_px & flood_ob))

    # Remove unreliable PW objects
    # print("{} - Removing unreliable PW areas...".format(dt.now()))
    if pw_selection == "bbk":
        pw_objects = measure.label(floodmap == l_pw, background=0)
        pw_props = measure.regionprops(pw_objects, pw == 1)
        pw_labels = np.array([el["label"] for el in pw_props])
        pw_water = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with bbk == pw
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc)])] = l_dl
        del pw_objects, pw_props, pw_labels, pw_water
    elif pw_selection == "bbk-risk":
        pw_objects = measure.label(floodmap == l_pw, background=0)
        pw_props = measure.regionprops(pw_objects, pw == 1)  # intensity = (pw == 1)
        pw_labels = np.array([el["label"] for el in pw_props])
        pw_water = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with bbk == pw
        pw_props = measure.regionprops(pw_objects, np.max(np.array([ogp, ogf]), axis=0) > 0)  # intensity = (max of ogp and ogf) > 0
        pw_og = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with og > 0
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc) & (pw_og < pw_perc)])] = l_dl
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc) & (pw_og >= pw_perc)])] = l_ltf
        del pw_objects, pw_props, pw_labels, pw_water, pw_og  # , pw_trees

    # Indicate significantly large pixel-based objects
    # print("{} - Indication of large pixel groups...".format(dt.now()))
    flood_onlypx = (floodmap == 0) & (flood_px != 0)
    flood_px_ids_array, _ = measure.label(flood_onlypx, background=0, connectivity=2, return_num=True)
    flood_px_props = measure.regionprops(flood_px_ids_array)
    flood_px_ids_list = np.array([el["label"] for el in flood_px_props])
    flood_px_sizes = np.array([el["area"] for el in flood_px_props])
    floodmap[(floodmap == 0) & np.isin(flood_px_ids_array, flood_px_ids_list[flood_px_sizes > mmu])] = l_pf
    del flood_onlypx, flood_px_ids_array, flood_px_props, flood_px_ids_list, flood_px_sizes

    # Filter PF objects
    if pf_selection == "none":
        floodmap[floodmap == l_pf] = l_dl
    elif pf_selection == "risk":
        pf_objects = measure.label(floodmap == l_pf, background=0)
        pf_props = measure.regionprops(pf_objects, np.max(np.array([ogp, ogf]), axis=0) > 0)  # intensity = (max of ogp and ogf) > 0
        pf_labels = np.array([el["label"] for el in pf_props])
        pf_og = np.array([el["mean_intensity"] for el in pf_props]) * 100  # % of pixels in region with og > 0
        floodmap[np.isin(pf_objects, pf_labels[pf_og < pw_perc])] = l_dl
        del pf_objects, pf_props, pf_labels, pf_og

    #  RG into FV areas
    # print("{} - Region growing into flooded vegetation areas...".format(dt.now()))
    if fv_seeds == "all":
        seeds = [l_of, l_pf, l_pw, l_fv]
    elif fv_seeds == "of":
        seeds = [l_of, l_fv]
    floodmap = rg(floodmap, seeds, l_fv, (inc_vv > t_incvv) & (inc_r > t_incr))

    #  RG into FF
    # print("{} - Region growing into flooded forest areas...".format(dt.now()))
    if "depth" in ff_inclusion:
        floodmap = rg_ff(floodmap, trees, ogp, ogf, odp, odf, [l_of], ff_inclusion=ff_inclusion, dem=dem)
    else:
        floodmap = rg_ff(floodmap, trees, ogp, ogf, odp, odf, [l_of], ff_inclusion=ff_inclusion)

    # Clean up
    # print("{} - Cleaning up map...".format(dt.now()))
    # Remove small objects
    floodmap = remove_smallobjects(floodmap, [l_of, l_pw, l_fv, l_ltf, l_pf], mmu=mmu_holes)
    # Fill holes
    floodmap = fill_holes(floodmap, flood_vv, flood_vh, ref_vv, ref_vh, t_vv, t_vh, [l_dl, l_if], mmu=mmu_holes)

    # Indicate forested areas where flood state is unsure
    # print("{} - Indication of forested, unsure areas...".format(dt.now()))
    floodmap[(floodmap == 0) & (trees == 1)] = l_if

    # Mask NoData areas
    floodmap[(np.isnan(flood_vv)) | (np.isnan(ref_vv)) | (trees == nd_pres)] = nd_map

    # Return output
    return floodmap


def apply_terrafloodpx(flood_vv, flood_vh, ref_vv, ref_vh, inc_vv, inc_r, trees, pw, ogp, ogf, odp, odf, dem, t_vv,
                       t_vh, t_incvv=3, t_incr=3, mmu_core=10, mmu_holes=5, pf_selection="all", ff_inclusion="all",
                       pw_selection="all", pw_perc=50, fv_seeds="all"):
    """
    Apply pixel-based version of TerraFlood algorithm
    
    Inputs
    flood_vv: nd array
        array of flood VV backscatter values
    flood_vh: nd array
        array of flood VH backscatter values
    ref_vv: nd array
        array of ref. VV backscatter values
    ref_vh: nd array
        array of ref. VH backscatter values
    inc_vv: nd array
        array of increase in VV backscatter values
    inc_r: nd array
        array of increase in (VV-VH) backscatter values
    trees: nd array, bool
        array indicating tree presence
    pw: nd array, bool
        array indicating permanent water presence
    ogp: nd array
        array indicating pluvial flood susceptibility classes
    ogf: nd array
        array indicating fluvial flood susceptibility classes
    odp: nd array
        array indicating pluvial flood depth classes
    odf: nd array
        array indicating fluvial flood depth classes
    dem: nd array
        array of elevation values
    t_vv: float
        VV threshold value
    t_vh: float
        VH threshold value
    t_incvv: float
        incVV threshold value
    t_incr: float
        incR threshold value
    mmu_core: int
        minimal mapping unit for core flooding
    mmu_holes: int
        minimal mappint unit for holes
    pf_selection: string
        how to filter the PF class, one of ["all", "risk", "none"]
    ff_inclusion: string
        how to include the FF class, one of ["all", "risk", "depth", "risk-depth"]
    pw_selection: string
        how to filter the PW class, one of ["all", "bbk", "bbk-risk"]
    pw_perc: int
        percentage value [0-100] used for PW/PF filtering
    fv_seeds: list
        list of flood classes to consider as seeds for FV region growing
    Outputs:
    floodmap: nd array
        TerraFlood flood map (classes defined on L35-42)
    """
    # Single band thresholding
    pwmap_vv = (flood_vv < t_vv) & (ref_vv < t_vv)
    pwmap_vh = (flood_vh < t_vh) & (ref_vh < t_vh)
    floodmap_vv = (flood_vv < t_vv) & (pwmap_vv == 0)
    floodmap_vh = (flood_vh < t_vh) & (pwmap_vh == 0)

    # Combine VV and VH maps
    # Get core
    pwmap_combo = apply_mmu(pwmap_vv & pwmap_vh, mmu_core)
    floodmap_combo = apply_mmu(floodmap_vv & floodmap_vh, mmu_core)
    # RG from core for PW
    pwmap_combo = rg(pwmap_combo, [1], 1, (pwmap_vv & pwmap_vh), selem=morph.selem.rectangle(3, 3))
    # RG from core for OF
    floodmap_vv_kind = (flood_vv < t_vv + 1) & (pwmap_vv == 0)
    floodmap_vh_kind = (flood_vh < t_vh + 1) & (pwmap_vh == 0)
    floodmap_combo = rg(floodmap_combo, [1], 1, (floodmap_vv & floodmap_vh_kind) | (floodmap_vv_kind & floodmap_vh),
                        selem=morph.selem.rectangle(3, 3))
    del floodmap_vv_kind, floodmap_vh_kind
    # Combine PW and OF
    floodmap = np.zeros(pwmap_combo.shape, dtype="int")
    floodmap[floodmap_combo] = l_of
    floodmap[pwmap_combo] = l_pw
    del pwmap_combo, floodmap_combo

    # Region grow into groups of pixels satisfying class definitions
    # print("{} - Region growing into PW areas...".format(dt.now()))
    floodmap = rg(floodmap, [l_pw, l_of], l_pw, (pwmap_vv & pwmap_vh))
    del pwmap_vh, pwmap_vv
    # print("{} - Region growing into OF areas...".format(dt.now()))
    floodmap = rg(floodmap, [l_pw, l_of], l_of, (floodmap_vv & floodmap_vh))
    del floodmap_vv, floodmap_vh

    # Apply closing to refine flood edges + remove small RG patches
    # print("{} - Refining edges and removing small objects...".format(dt.now()))
    pwmap_combo = remove_smallobjects(morph.binary_closing(floodmap == l_pw, selem=morph.selem.disk(1)), [1], mmu=2)
    floodmap_combo = remove_smallobjects(morph.binary_closing(floodmap == l_of, selem=morph.selem.disk(1)), [1], mmu=2)
    floodmap = np.zeros(pwmap_combo.shape, dtype="int")
    floodmap[floodmap_combo] = l_of
    floodmap[pwmap_combo] = l_pw

    # Remove unreliable PW objects
    # print("{} - Removing unreliable PW areas...".format(dt.now()))
    if pw_selection == "bbk":
        pw_objects = measure.label(floodmap == l_pw, background=0)
        pw_props = measure.regionprops(pw_objects, pw == 1)
        pw_labels = np.array([el["label"] for el in pw_props])
        pw_water = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with bbk == pw
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc)])] = l_dl
        del pw_objects, pw_props, pw_labels, pw_water
    elif pw_selection == "bbk-risk":
        pw_objects = measure.label(floodmap == l_pw, background=0)
        pw_props = measure.regionprops(pw_objects, pw == 1)
        pw_labels = np.array([el["label"] for el in pw_props])
        pw_water = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with bbk == pw
        pw_props = measure.regionprops(pw_objects, np.max(np.array([ogp, ogf]), axis=0) > 0)  # intensity = (max of ogp and ogf) > 0
        pw_og = np.array([el["mean_intensity"] for el in pw_props]) * 100  # % of pixels in region with og > 0
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc) & (pw_og < pw_perc)])] = l_dl
        floodmap[np.isin(pw_objects, pw_labels[(pw_water < pw_perc) & (pw_og >= pw_perc)])] = l_ltf
        del pw_objects, pw_props, pw_labels, pw_water, pw_og  # , pw_trees

    # Region grow into small OF objects
    # print("{} - Region growing into small OF areas...".format(dt.now()))
    seed_labels = [l_of, l_pw, l_pf]
    grow_criterium = ((flood_vv < t_vv + 1) & (flood_vh < t_vh)) | ((flood_vv < t_vv) & (flood_vh < t_vh + 1))
    floodmap = rg(floodmap, seed_labels, l_pf, grow_criterium=grow_criterium)
    del grow_criterium

    # Filter PF objects
    # print("{} - Filtering PF objects...".format(dt.now()))
    if pf_selection == "none":
        floodmap[floodmap == l_pf] = l_dl
    elif pf_selection == "risk":
        pf_objects = measure.label(floodmap == l_pf, background=0)
        pf_props = measure.regionprops(pf_objects, np.max(np.array([ogp, ogf]), axis=0) > 0)  # intensity = (max of ogp and ogf) > 0
        pf_labels = np.array([el["label"] for el in pf_props])
        pf_og = np.array([el["mean_intensity"] for el in pf_props]) * 100  # % of pixels in region with og > 0
        floodmap[np.isin(pf_objects, pf_labels[pf_og < pw_perc])] = l_dl
        del pf_objects, pf_props, pf_labels, pf_og

    #  RG into FV
    # print("{} - Region growing into flooded vegetation areas...".format(dt.now()))
    if fv_seeds == "all":
        seeds = [l_of, l_pf, l_pw, l_fv]
    elif fv_seeds == "of":
        seeds = [l_of, l_fv]
    floodmap = rg(floodmap, seeds, l_fv, (inc_vv > t_incvv) & (inc_r > t_incr))

    #  RG into FF
    # print("{} - Region growing into flooded forest areas...".format(dt.now()))
    if "depth" in ff_inclusion:
        floodmap = rg_ff(floodmap, trees, ogp, ogf, odp, odf, [l_of], ff_inclusion=ff_inclusion, dem=dem)
    else:
        floodmap = rg_ff(floodmap, trees, ogp, ogf, odp, odf, [l_of], ff_inclusion=ff_inclusion)

    # Clean up
    # Remove small objects (pw, of, fv, pf)
    # print("{} - Cleaning up map: removing small objects...".format(dt.now()))
    floodmap = remove_smallobjects(floodmap, [l_of, l_pw, l_fv, l_ltf, l_pf], mmu=mmu_holes)
    # Final closing to refine edges
    # print("{} - Cleaning up map: refining edges...".format(dt.now()))
    floodmap_old = np.copy(floodmap)
    for l_class in [l_pw, l_of, l_pf, l_fv]:  # order here determines priority!
        floodmap_class = morph.binary_closing(floodmap_old == l_class, selem=morph.selem.disk(1))
        floodmap[(floodmap == 0) & (floodmap_class == 1)] = l_class
    del floodmap_old
    # Fill holes
    # print("{} - Cleaning up map: filling holes...".format(dt.now()))
    floodmap = fill_holes(floodmap, flood_vv, flood_vh, ref_vv, ref_vh, t_vv, t_vh, class_labels_holes=[l_dl, l_if],
                          mmu=mmu_holes)

    # Indicate forested areas where floodmap state is unsure
    # print("{} - Indication of forested, unsure areas...".format(dt.now()))
    floodmap[(floodmap == 0) & (trees == 1)] = l_if

    # Mask NoData areas
    floodmap[(np.isnan(flood_vv)) | (np.isnan(ref_vv)) | (trees == nd_pres)] = nd_map

    # Return output
    return floodmap


def combine_floodmaps(floodmap1, floodmap2, pw, trees, pf_selection="all", ogp=None, ogf=None, mmu=5, pw_perc=50):
    """
    Combine pixel and object based TerraFlood maps
    
    Inputs
    floodmap1: nd array
        first array of flood classes
    floodmap2: nd array
        second array of flood classes
    pw: nd array, bool
        array indicating permanent water presence
    trees: nd array, bool
        array indicating tree presence
    pf_selection: string
        how to filter PF class, one of ["all", "risk", "none"]
    ogp: nd array
        array indicating pluvial flood susceptibility classes
    ogf: nd array
        array indicating fluvial flood susceptibility classes
    mmu: int
        minimal mapping unit for core flooding
    pw_perc: int
        percentage value [0-100] to filter PF class
    Outputs:
    floodmap: nd array
        TerraFlood flood map (classes defined on L35-42)
    """
    flood_footprint = combine_footprints(np.isin(floodmap1, [l_pw, l_of, l_ltf, l_pf, l_fv]),
                                         np.isin(floodmap2, [l_pw, l_of, l_ltf, l_pf, l_fv]), mmu_core=mmu)  # select union + all patches overlapping union
    floodmap_combo = np.zeros(flood_footprint.shape)
    floodmap_combo[flood_footprint & (floodmap1 == l_pw) & (floodmap2 == l_pw)] = l_pw
    floodmap_combo[flood_footprint & (floodmap1 == l_of) & (floodmap2 == l_of)] = l_of
    floodmap_combo[flood_footprint & (floodmap1 == l_fv) & (floodmap2 == l_fv)] = l_fv
    floodmap_combo[flood_footprint & (floodmap1 == l_pf) & (floodmap2 == l_pf)] = l_pf
    floodmap_combo[flood_footprint & (floodmap1 == l_ltf) & (floodmap2 == l_ltf)] = l_ltf

    floodmap_combo[flood_footprint & (((floodmap1 == l_pw) & (floodmap2 == l_of)) |
                                      ((floodmap1 == l_of) & (floodmap2 == l_pw)))] = l_of
    floodmap_combo[flood_footprint & (((floodmap1 == l_pw) & (floodmap2 == l_ltf)) |
                                      ((floodmap1 == l_ltf) & (floodmap2 == l_pw)))] = l_ltf
    floodmap_combo[flood_footprint & (((floodmap1 == l_pw) & (floodmap2 == l_fv)) |
                                      ((floodmap1 == l_fv) & (floodmap2 == l_pw)))] = l_pw
    floodmap_combo[flood_footprint & (((floodmap1 == l_pw) & (floodmap2 == l_pf)) |
                                      ((floodmap1 == l_pf) & (floodmap2 == l_pw)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_pw) & np.isin(floodmap2, [l_dl, l_ff, l_if])) |
                                      (np.isin(floodmap1, [l_dl, l_ff, l_if]) & (floodmap2 == l_pw)))] = l_pw

    floodmap_combo[flood_footprint & (((floodmap1 == l_of) & (floodmap2 == l_ltf)) |
                                      ((floodmap1 == l_ltf) & (floodmap2 == l_of)))] = l_of
    floodmap_combo[flood_footprint & (((floodmap1 == l_of) & (floodmap2 == l_fv)) |
                                      ((floodmap1 == l_fv) & (floodmap2 == l_of)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_of) & (floodmap2 == l_pf)) |
                                      ((floodmap1 == l_pf) & (floodmap2 == l_of)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_of) & np.isin(floodmap2, [l_dl, l_ff, l_if])) |
                                      (np.isin(floodmap1, [l_dl, l_ff, l_if]) & (floodmap2 == l_of)))] = l_pf

    floodmap_combo[flood_footprint & (((floodmap1 == l_ltf) & (floodmap2 == l_fv)) |
                                      ((floodmap1 == l_fv) & (floodmap2 == l_ltf)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_ltf) & (floodmap2 == l_pf)) |
                                      ((floodmap1 == l_pf) & (floodmap2 == l_ltf)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_ltf) & np.isin(floodmap2, [l_dl, l_ff, l_if])) |
                                      (np.isin(floodmap1, [l_dl, l_ff, l_if]) & (floodmap2 == l_ltf)))] = l_pf

    floodmap_combo[flood_footprint & (((floodmap1 == l_fv) & (floodmap2 == l_pf)) |
                                      ((floodmap1 == l_pf) & (floodmap2 == l_fv)))] = l_pf
    floodmap_combo[flood_footprint & (((floodmap1 == l_fv) & np.isin(floodmap2, [l_dl, l_ff, l_if])) |
                                      (np.isin(floodmap1, [l_dl, l_ff, l_if]) & (floodmap2 == l_fv)))] = l_fv

    floodmap_combo[flood_footprint & (((floodmap1 == l_pf) & np.isin(floodmap2, [l_dl, l_ff, l_if])) |
                                      (np.isin(floodmap1, [l_dl, l_ff, l_if]) & (floodmap2 == l_pf)))] = l_pf

    floodmap_combo[(floodmap1 == l_ff) & (floodmap2 == l_ff)] = l_ff

    floodmap_combo[~flood_footprint & (np.isin(floodmap1, [l_pw, l_of, l_ltf, l_fv, l_pf]) |
                                       np.isin(floodmap2, [l_pw, l_of, l_ltf, l_fv, l_pf]))] = l_pf
                                       
    # Filter PF objects
    if pf_selection == "none":
        floodmap_combo[floodmap_combo == l_pf] = l_dl
    elif pf_selection == "risk":
        pf_objects = measure.label(floodmap_combo == l_pf, background=0)
        pf_props = measure.regionprops(pf_objects, np.max(np.array([ogp, ogf]), axis=0) > 0)  # intensity = (max of ogp and ogf) > 0
        pf_labels = np.array([el["label"] for el in pf_props])
        pf_og = np.array([el["mean_intensity"] for el in pf_props]) * 100  # % of pixels in region with og > 0
        pf_props = measure.regionprops(pf_objects, pw == 1)  # intensity = pw == 1
        pf_pw = np.array([el["mean_intensity"] for el in pf_props]) * 100  # % of pixels in region with pw == 1
        floodmap_combo[np.isin(pf_objects, pf_labels[pf_og < pw_perc])] = l_dl
        floodmap_combo[np.isin(pf_objects, pf_labels[pf_pw >= pw_perc])] = l_pw
        del pf_objects, pf_props, pf_labels, pf_og

    # Remove isolated FV objects
    flood_footprint = np.isin(floodmap_combo, [l_of])
    flood_footprint_dilated = morph.binary_dilation(flood_footprint == 1, selem=morph.selem.disk(1))
    fv_objects = measure.label(floodmap_combo == l_fv, background=0)
    fv_props = measure.regionprops(fv_objects, flood_footprint_dilated)  # intensity = flood footprint
    fv_labels = np.array([el["label"] for el in fv_props])
    fv_footprint = np.array([el["mean_intensity"] for el in fv_props]) * 100  # % of pixels in region within dilated footprint
    floodmap_combo[np.isin(fv_objects, fv_labels[fv_footprint == 0])] = l_dl
    # print("Removed {} isolated FV objects".format(np.sum(np.isin(fv_objects, fv_labels[fv_footprint == 0]))))
    
    # Remove isolated FF objects
    flood_footprint = apply_mmu(np.isin(floodmap_combo, [l_of]), mmu=mmu, background=0, connectivity=2)  # mmu parameter set to 5
    flood_footprint_dilated = morph.binary_dilation(flood_footprint == 1, selem=morph.selem.disk(1))
    ff_objects = measure.label(floodmap_combo == l_ff, background=0)
    ff_props = measure.regionprops(ff_objects, flood_footprint_dilated)  # intensity = flood footprint
    ff_labels = np.array([el["label"] for el in ff_props])
    ff_footprint = np.array([el["mean_intensity"] for el in ff_props]) * 100  # % of pixels in region within dilated footprint
    floodmap_combo[np.isin(ff_objects, ff_labels[ff_footprint == 0])] = l_dl
    # print("Removed {} isolated FF objects".format(np.sum(np.isin(ff_objects, ff_labels[ff_footprint == 0]))))

    # Clean up
    floodmap_combo = remove_smallobjects(floodmap_combo, [l_pw, l_of, l_fv, l_ltf, l_pf], mmu=mmu)

    # Indicate forested areas where floodmap state is unsure
    floodmap_combo[(floodmap_combo == l_dl) & (trees == 1)] = l_if
    
    # Mask NoData areas
    floodmap_combo[(floodmap1 == nd_map) | (floodmap2 == nd_map)] = nd_map

    return floodmap_combo
