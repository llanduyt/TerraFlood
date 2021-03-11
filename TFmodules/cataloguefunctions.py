# -*- coding: utf-8 -*-
"""
@author: llanduyt
@author: Lisa Landuyt
@purpose: Ancillary functions to find and handle data
    Based on https://github.com/VITObelgium/notebook-samples/blob/master/Terrascope/Advanced/OpenSearchDemoFinal.ipynb
"""

import os
import datetime
import requests
import shapely
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
from rasterio import warp
from rasterio import fill
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.crs import CRS
from pyproj import Proj
from pyproj import transform as projtransform

wgs = "epsg:4326"
es_ops_url = 'https://services.terrascope.be/catalogue/'  # ops environment
producttype_s1 = "urn:eop:VITO:CGS_S1_GRD_SIGMA0_L1"
producttype_s2 = "urn:eop:VITO:TERRASCOPE_S2_TOC_V2"


def find_products(urn, elastic_search_url=es_ops_url, start=datetime.datetime(2015, 1, 1, 0, 0, 0).isoformat(),
                  end=datetime.datetime(2030, 12, 31, 23, 59, 59).isoformat(), lonmin=-180.0, latmin=-90.0,
                  lonmax=180.0, latmax=90.0, ccmin=0.0, ccmax=100.0,
                  prstart=datetime.datetime(2015, 1, 1, 0, 0, 0).isoformat(),
                  prend=datetime.datetime(2021, 12, 31, 23, 59, 59).isoformat(),
                  t_id='', on_terrascope=True, print_url=False):
    
    """
    Find products in the Terrascope catalogue.

    Inputs
    urn: str
        urn of the collection
    elastic_search_url: str
        ops environment (default = es_ops_url = "https://services.terrascope.be/catalogue/")
    start: str
        start acquisition datetime in ISO format (default = "2015-01-01")
    end: str
        end acquisition datetime in ISO format (default = "2030-12-31")
    prstart: str
        start datetime of product publication in ISO format (default = "2015-01-01")
    prend: 
        end datetime of product publication in ISO format (default = "2030-12-31")
    latmin: float
        minimum latitude of bounding box in degrees (default = -90.0)
    latmax: float
        maximum latitude of bounding box in degrees (default = 90.0)
    lonmin: float
        minimum longitude of bounding box in degrees (default = -180.0)
    lonmax: float 
        maximum longitude of bounding box in degrees (default = 180.0)
    ccmin: float
        Sentinel-2 minimum cloud cover in percent (not used for Sentinel-1 queries) (default = 0.0)
    ccmax: float
        Sentinel-2 maximum cloud cover in percent (not used for Sentinel-1 queries) (default = 100.0)
    t_id: str
        Sentinel-2 tile ID (not used for Sentinel-1 queries) (default = "")
    on_terrascope: bool
        if True, then the function returns the path to the data; if False, then a download hyprlink is returned (default = True)
    print_url: bool 
        will print out the query URL (default = False)
    
    Ouputs
    productlist: list of product dictionaries containing product information; keys:
        productID: a unique product identifier, composed of the collection's urn and the productTitle
        bbox: bounding box coordinates, in this order: longitude min, latitude min, longitude max, latitude max
        productDate: acquisition starting datetime of the product
        productPublishedDate: datetime of publication of the product in the catalog
        relativeOrbit: acquisitions with the same relative orbit overlap completely
        productType: the product's description
        cloudcover: percentage of clouds in the product (Sentinel-2 only)
        tileID: product identifier in the UTM tiling grid (Sentinel-2 only)
        files: a list of files associated to this product; keys:
            filetype: 
                preview: a quicklook image
                data: the actual data
                related: data derivered from the actual data
                alternates: metadata
            filepath: actual string that points to the file on the Terrascope file system
            category: file type qualifier
            title: file content description
            length: file size in bytes
    """

    products_list = []
    bbox = str(lonmin)+','+str(latmin)+','+str(lonmax)+','+str(latmax)
    requestbasestring = elastic_search_url + 'products?collection=' + urn + "&start=" + str(start) + "&end=" + \
        str(end) + '&bbox=' + bbox
    requestbasestring = requestbasestring + '&modificationDate=[' + str(prstart) + ',' + str(prend) + "["
    
    if 'S2' in urn:  # cloud cover is not relevant for S1 products
        requestbasestring = requestbasestring + '&cloudCover=['+str(ccmin)+','+str(ccmax)+']'
        if t_id != '':  # there are tile IDs only for S2 products
            requestbasestring = requestbasestring + '&sortKeys=title,,0,0&tileId=' + t_id
        else:
            requestbasestring = requestbasestring + '&sortKeys=title,,0,0'
    if on_terrascope:
        requestbasestring = requestbasestring + '&accessedFrom=MEP'
    if print_url:
        print(requestbasestring+'&startIndex=1')  # printing this is useful if you want to paste it in a browser
    
    products = requests.get(requestbasestring+'&startIndex=1')
    products_json = products.json()
    num_products = products_json['totalResults']
    print(str(num_products) + ' products found between ' + str(start) + ' and ' + str(end) + ' produced between '
          + str(prstart) + ' and ' + str(prend))
    items_per_page = int(products_json['itemsPerPage'])
    
    if num_products > 10000:
        print('too many results (max 10000 allowed), please narrow down your search')
        return ['too many results']
    else:    
        if num_products > 0:
            for ind in range(int(num_products / items_per_page) + 1):
                startindex = ind * items_per_page + 1
                products = requests.get(requestbasestring+'&startIndex=' + str(startindex))
                products_json = products.json()
                features = products_json['features']
                
                for f in features:
                    productdetail = {}
                    productdetail['productID'] = f['id']
                    productdetail['bbox'] = f['bbox']
                    productdetail['productDate'] = f['properties']['date']
                    productdetail['productPublishedDate'] = f['properties']['published']
                    productdetail['productTitle'] = f['properties']['title']
                    productdetail['relativeOrbit'] = \
                        f['properties']['acquisitionInformation'][1]['acquisitionParameters']['relativeOrbitNumber']
                    productdetail['productType'] = f['properties']['productInformation']['productType']
                    
                    if 'S2' in f['id']:
                        productdetail['cloudcover'] = f['properties']['productInformation']['cloudCover']
                        productdetail['tileID'] = \
                            f['properties']['acquisitionInformation'][1]['acquisitionParameters']['tileId']
                    else:
                        productdetail['cloudcover'] = ''
                        productdetail['tileID'] = ''
                    
                    filelist = []
                    linkkeys = f['properties']['links'].keys()
                    
                    for linkkey in linkkeys:
                        for fil in f['properties']['links'][linkkey]:
                            filedetails = {}
                            filedetails['filetype'] = linkkey
                            
                            if on_terrascope:
                                filedetails['filepath'] = fil['href'][7:]
                            else:
                                filedetails['filepath'] = fil['href']
                            
                            if linkkey == 'previews':
                                filedetails['category'] = fil['category']
                                filedetails['title'] = fil['category']
                            
                            if (linkkey == 'alternates') | (linkkey == 'data'):
                                filedetails['category'] = fil['title']
                                filedetails['title'] = fil['title']
                            
                            if linkkey == 'related':
                                filedetails['category'] = fil['category']
                                filedetails['title'] = fil['title']
                            filedetails['length'] = fil['length']
                            filelist.append(filedetails)
                            
                    productdetail['files'] = filelist        
                    products_list.append(productdetail)
            return products_list
        else:
            return ['no products']


def find_s1_ref(floodim_timestamp, lon_min, lat_min, lon_max, lat_max, rel_orbit, floodim_bbox, deltat=12,
                min_coverage=100):
    """
    Find S1 reference image matching flood image
    
    Inputs
    floodim_timestamp: datetime object
        acquisition time of flood image
    lon_min, lat_min, lon_max, lat_max: float
        coordinates of ROI
    rel_orbit: float
        rel. orbit of flood image
    floodim_bbox: list
        bounding box of flood image
    deltat: int
        number of days between flood and ref. image (multitude of 6)
    min_coverage: int
        minimal coverage (in percentage, 0-100) of ROI by ref. image
    Outputs
    p_ref: dict
        reference product
    """
    i_max = 5  # max. number of orbit cycles to go back in time
    startdate = floodim_timestamp - datetime.timedelta(days=deltat+1)
    enddate = startdate + datetime.timedelta(days=2)
    # Find products
    ref_found = False
    dt_attempted = []
    i = 0
    while (not ref_found) & (i < i_max):
        products = find_products(urn=producttype_s1, start=startdate.isoformat(), end=enddate.isoformat(),
                                 latmin=lat_min, latmax=lat_max, lonmin=lon_min, lonmax=lon_max)
        if products[0] != "no products":
            products = [p for p in products if p["relativeOrbit"] == rel_orbit]
            if len(products) > 0:
                # Select products covering ROI minimally
                dc = np.zeros(len(products))
                for i_p, product in enumerate(products):
                    vv_filepath = [el["filepath"] for el in product["files"] if el["title"] == "VV"][0]
                    w = get_window(vv_filepath, lon_min, lat_min, lon_max, lat_max, wgs)[0]
                    dc[i_p] = get_datacoverage(vv_filepath, window=w)
                if np.sum(dc > min_coverage) == 0:
                    print("Error! No products found satisfying minimal data coverage.")
                    products = [p for i_p, p in enumerate(products) if dc[i_p] > 0]
                else:
                    products = [p for i_p, p in enumerate(products) if dc[i_p] > min_coverage]
                if len(products) > 0:
                    # Select product with largest overlap bbox (same slice)
                    floodim_bbox_geom = shapely.geometry.box(*tuple(floodim_bbox))
                    overlap_bbox = np.array(
                        [floodim_bbox_geom.intersection(shapely.geometry.box(*tuple(product["bbox"]))).area
                         for product in products])
                    ref_found = True
        if not ref_found:
            dt_attempted.append(deltat)
            deltat_old = deltat
            while (deltat in dt_attempted) & (i < i_max):
                i += 1
                deltat = i * 6
            print("Warning! No products found {} days prior to flood date, swapping to {} days.".format(deltat_old,
                                                                                                        deltat))
            startdate = floodim_timestamp - datetime.timedelta(days=deltat + 1)
            enddate = startdate + datetime.timedelta(days=2)
    if (not ref_found) & (len(products) == 0):
        print("Warning! No products found with requested rel. orbit {} and min. data coverage".format(rel_orbit))
        return "no products"
    elif (not ref_found) & (products[0] == "no products"):
        print("Error! No products found within {} orbit cycles preceding flood date.".format(i_max))
        return "no products"
    else:
        return products[np.argmax(overlap_bbox)]


def tif2ar(infile, window=None, band=None, return_bandnames=False):
    """ Read in a .tif file to a numpy array

    Inputs:
    infile: str
        Path + name of input .tif file
    window: pair of tuples
        ((rmin, rmax), (cmin, cmax)) defining the indices of the columns/rows to read
    band: int
        Index (starting from 1) of the band to read. If None, all bands are read
    return_bandnames: bool
        If true, a list containing the band descriptions (if set) will be returned
    Outputs:
    array: nd array
        Image pixel values
    metadata: dict
        Dictionary containing metadata: driver, dtype, nodata, width, height, count, crs, transform
    band_names (if return_bandnames is True): list
        List of band descriptions
    """
    if not os.path.isfile(infile):
        print("Error! File {} does not exist.".format(infile))
        if return_bandnames:
            return None, None, None
        return None, None
    with rasterio.open(infile) as src:
        metadata = src.meta
        if src.count == 1:
            band = 1
        array = src.read(band, window=window)
        if return_bandnames:
            band_names = list(src.descriptions)
    if return_bandnames:
        return array, metadata, band_names
    return array, metadata


def ar2tif(array, outfile, crs, transform, dtype=rasterio.float32, band_index=0, nodata_value=None, band_names=None):
    """ Export a numpy array to an GeoTiff

    Inputs:
    array: nd array
       Image pixel values. Should have shape (rows, cols, bands). If not, specify the band_index.
    outfile : str
       Path + name of the output .tif file.
    crs: rasterio.crs.CRS object or int
        CRS object or EPSG code of crs.
    transform: Affine
        Transformation from pixel to geographic coordinates.
    dtype: rasterio dtype or None (default=rasterio.float32)
       If None, equal to the raster dtype.
    band_index: int (default=0)
        Index of bands in the shape tuple. Possible values: 0, 2.
    nodata_value: int/float
        Value for no data
    band_names: list or None (default=None)
        List of descriptions for the bands.
    """

    s = array.shape
    if len(s) == 2:
        rows, cols = s
        bands = 1
    elif len(s) == 3:
        if band_index == 2:
            rows, cols, bands = s
        elif band_index == 0:
            bands, rows, cols = s
        else:
            print("Error! Band_index value invalid. Should be 0 or 2, is {}. Aborting.".format(band_index))
    else:
        bands = None
        print("Error! Array shape invalid. Aborting.")
    if bands is not None:
        if band_names and len(band_names) != bands:
            print("Error! List band_names should have length equal to no. of bands. Currently {} vs. {}. "
                  "No band names saved.".format(len(band_names), bands))
            band_names = None
    
        if str(np.dtype(array.dtype)) != dtype:
            dtype = str(np.dtype(array.dtype))
    
        metadata = {
            "driver": 'GTiff',
            "height": rows,
            "width": cols,
            "count": bands,
            "crs": crs,
            "dtype": dtype,
            "transform": transform
        }
        if nodata_value is not None:
            metadata["nodata"] = nodata_value
        with rasterio.open(outfile, 'w', **metadata) as dst:
            if bands == 1:
                dst.write(array, 1)
                if band_names is not None:
                    dst.set_band_description(1, band_names[0])
            else:
                for b in range(bands):
                    if band_index == 2:
                        dst.write_band(b + 1, array[:, :, b])
                    elif band_index == 0:
                        dst.write_band(b + 1, array[b, :, :])
                    if band_names is not None:
                        dst.set_band_description(b + 1, band_names[b])


def get_mask_unique_values(src_filename, window=None, return_counts=False):
    """ 
    Get unique values of src_filename in window 
    
    Inputs
    src_filename: str
        path to source file
    window: rasterio Window
        window for which to extract unique values
    return_counts: bool
        whether to return counts of unique values
    """
    with rasterio.open(src_filename) as src:
        m = src.read_masks(window=window)
    return np.unique(m, return_counts=return_counts)


def get_datacoverage(src_filename, nodata_value=0, window=None):
    """ 
    Get the percentage of the window for which src_filename contains valid data
    
    Inputs
    src_filename: str
        path to source file
    nodata_value: int/float
        value representing no data
    window: rasterio Window
        window for which to extract unique values
    """
    u, c = get_mask_unique_values(src_filename, window=window, return_counts=True)
    if nodata_value in u:
        return (1 - c[u == nodata_value][0] / np.sum(c)) * 100
    return 100


def get_window(src_filename, xmin, ymin, xmax, ymax, crs_limits_epsg):
    """Get window of (xmin, ymin, xmax, ymax) in src_filename

    Inputs:
    src_filename: str
        path to source file
    xmin, ymin, xmax, ymax: floats
        coordinates describing ROI
    crs_limits_epsg: str
        CSR in which limits are expressed (in format "EPSG:code")
    Outputs:
        w: window
        coverage: int
            whether coverage is non-existing (-1), partial (0) or full (1)
    """
    with rasterio.open(src_filename) as src:
        west, south = projtransform(Proj(init=crs_limits_epsg), Proj(src.crs.to_proj4()), xmin, ymin)
        east, north = projtransform(Proj(init=crs_limits_epsg), Proj(src.crs.to_proj4()), xmax, ymax)
        rmin, cmin = src.index(west, north)
        rmax, cmax = src.index(east, south)
        width = src.width
        height = src.height
    if (rmax < 0) or (cmax < 0) or (rmin > height) or (cmin > width):
        coverage = -1
    elif (rmin < 0) or (cmin < 0) or (rmax > height) or (cmax > width):
        coverage = 0
    else:
        coverage = 1
    if coverage < 1:
        w = ((rmin, rmax), (cmin, cmax))
    else:
        w = Window.from_slices((rmin, rmax), (cmin, cmax))
    return w, coverage


def find_tiles_kbl(roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_crs, kbl_dir, kbl_scale=0):
    """
    Find KBL tiles covered by window [roi_xmin, roi_ymin, roi_xmax, roi_ymax]
    
    Inputs
    roi_xmin, roi_ymin, roi_xmax, roi_ymax: float
        coordinates of ROI
    roi_crs: str
        CRS in which ROI limits are expressed (format "EPSG:code")
    kbl_scale: int
        KBL scale (0, 4, 8, 16)
    Outputs
    tiles_id: list
        list of KBL tile IDs
    """
    roi_footprint = shapely.geometry.box(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
    if kbl_scale != 0:
        kbl_file = os.path.join(kbl_dir, "Kbl{}.shp".format(kbl_scale))
    else:
        kbl_file = os.path.join(kbl_dir, "Kbl.shp")
    if not os.path.exists(kbl_file):
        print("Error! Kbl file not found: {}".format(kbl_file))
    kbl = gpd.read_file(kbl_file).to_crs(roi_crs)
    tiles_id = kbl.loc[kbl.geometry.intersects(roi_footprint), "CODE"]
    return list(tiles_id)


def mosaic_tiles(dir_tiles, roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_transform, roi_width, roi_length, roi_crs,
                 tile_dtype, nodata_value, kbl_dir, kbl_scale=0, query_string="", resampling="nearest", fill_nodata=True,
                 mosaic_filename=None):
    """
    Combine KBL tiles into mosaic
    
    Inputs
    dir_tiles: str
        directory to image tiles
    roi_xmin, roi_ymin, roi_xmax, roi_ymax: float
        coordinates of ROI 
    roi_transform: rasterio transform
        ROI transform
    roi_width: int
        ROI width (no. of columns)
    roi_length: int
        ROI length (no. of rows)
    roi_crs: str
        CRS in which ROI limits are expressed (format "EPSG:code")
    tile_dtype: numpy dtype
        mosaic datatype
    nodata_value: int/float 
        value for no data
    kbl_scale: int 
        KBL scale (0, 8, 16)
    query_string: str
        query string to search for tile files
    resampling: str
        resampling mode for warp_to_transform
    fill_nodata: bool
        whether to fill no data areas
    mosaic_filename: str
        path to store mosaic
    
    Outputs
    mos: nd array
        mosaic image
    """
    tiles_id = find_tiles_kbl(roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_crs, kbl_dir, kbl_scale=kbl_scale)
    if kbl_scale == 0:  # TODO: add for kbl_scale 4
        tiles = [glob(os.path.join(dir_tiles, "{}*{:02d}.tif".format(query_string, int(i)))) for i in tiles_id]
    elif kbl_scale == 8:
        tiles = [glob(os.path.join(dir_tiles, "{}*{:03d}.tif".format(query_string, int(i.replace("/", "")))))
                 for i in tiles_id]
    elif kbl_scale == 16:
        tiles = [glob(os.path.join(dir_tiles, "{}*{:03d}{}.tif".format(query_string, int(i[:-1].replace("/", "")),
                                                                       i[-1].lower()))) for i in tiles_id]
    tiles = [val for sublist in tiles for val in sublist]
    if len(tiles) == 0:
        print("Error! No tiles covering ROI found.")
        return None
    mos = np.ones((roi_length, roi_width), dtype=tile_dtype) * nodata_value
    for tile in tiles:
        tile_arr, tile_nd = warp_to_transform(os.path.join(dir_tiles, tile), roi_transform, roi_width,
                                              roi_length, roi_crs, return_nodata=True, resampling=resampling)
        tile_arr[tile_arr == tile_nd] = nodata_value
        mos[(mos == nodata_value) & (tile_arr != tile_nd)] = tile_arr[
            (mos == nodata_value) & (tile_arr != tile_nd)]
    if fill_nodata & (np.sum(mos == nodata_value) > 0):
        mos = fill.fillnodata(mos, mos != nodata_value, max_search_distance=100.0, smoothing_iterations=0)
    if mosaic_filename is not None:
        ar2tif(mos, mosaic_filename, roi_crs, roi_transform, dtype=mos.dtype, band_names=["LC"], 
               nodata_value=nodata_value)
    return mos


def time_to_utc(timestamp):
    """Transform time from local Belgian time (winter/summer) to UTC"""
    if not pd.isnull(timestamp):
        if timestamp.year == 2015:
            day_change_summer = 29
            day_change_winter = 25
        elif timestamp.year == 2016:
            day_change_summer = 27
            day_change_winter = 30
        elif timestamp.year == 2017:
            day_change_summer = 26
            day_change_winter = 29
        elif timestamp.year == 2018:
            day_change_summer = 25
            day_change_winter = 28
        elif timestamp.year == 2019:
            day_change_summer = 31
            day_change_winter = 27
        elif timestamp.year == 2020:
            day_change_summer = 29
            day_change_winter = 25
        elif timestamp.year == 2021:
            day_change_summer = 28
            day_change_winter = 31
        elif timestamp.year == 2022:
            day_change_summer = 27
            day_change_winter = 30
        else:
            print("No summer/winter time changes set for year {}. Sticking to UTC+1.".format(timestamp.year))
            return timestamp + datetime.timedelta(hours=1)
        if ((timestamp.month > 3) and (timestamp.month < 10)) or \
                ((timestamp.month == 3) and (timestamp.day >= day_change_summer)) or \
                ((timestamp.month == 10) and (timestamp.day < day_change_winter)):
            return timestamp - datetime.timedelta(hours=2)
        else:
            return timestamp - datetime.timedelta(hours=1)
    return timestamp


def time_to_local(timestamp):
    """Transform time from UTC to local Belgian time (winter/summer)"""
    if not pd.isnull(timestamp):
        if timestamp.year == 2015:
            day_change_summer = 29
            day_change_winter = 25
        elif timestamp.year == 2016:
            day_change_summer = 27
            day_change_winter = 30
        elif timestamp.year == 2017:
            day_change_summer = 26
            day_change_winter = 29
        elif timestamp.year == 2018:
            day_change_summer = 25
            day_change_winter = 28
        elif timestamp.year == 2019:
            day_change_summer = 31
            day_change_winter = 27
        elif timestamp.year == 2020:
            day_change_summer = 29
            day_change_winter = 25
        elif timestamp.year == 2021:
            day_change_summer = 28
            day_change_winter = 31
        elif timestamp.year == 2022:
            day_change_summer = 27
            day_change_winter = 30
        else:
            print("No summer/winter time changes set for year {}. Sticking to UTC+1.".format(timestamp.year))
            return timestamp + datetime.timedelta(hours=1)
        if ((timestamp.month > 3) and (timestamp.month < 10)) or \
                ((timestamp.month == 3) and (timestamp.day >= day_change_summer)) or \
                ((timestamp.month == 10) and (timestamp.day < day_change_winter)):
            return timestamp + datetime.timedelta(hours=2)
        else:
            return timestamp + datetime.timedelta(hours=1)
    return timestamp


def warp_to_transform(src_filename, dst_transform, dst_width, dst_height, dst_crs_init, src_band=1, return_nodata=False,
                      resampling="nearest"):
    """ 
    Warp content of src_filename to match transform
    
    Inputs
    src_filename: str
        path to source file
    dst_transform: rasterio transform
        destination transform
    dst_width: int
        destination width (no. of columns)
    dst_height: int
        destination height (no. of rows)
    dst_crs_init: str
        destination CRS (format "EPSG:code")
    src_band: int
        index of band to warp, starting from 1 (default = 1)
    return_nodata: bool
        whether to return no data value (default = False)
    resampling: str
        resampling mode, one of ["nearest", "average", "cubic", "bilinear", "min", "max"]
    Outputs
    dst_array: nd array
        warped array
    nd_value: float
        no data value warped array (onli if return_nodata = True)
    """
    if resampling == "nearest":
        resampling = Resampling.nearest
    elif resampling == "average":
        resampling = Resampling.average
    elif resampling == "cubic":
        resampling = Resampling.cubic
    elif resampling == "bilinear":
        resampling = Resampling.bilinear
    elif resampling == "min":
        resampling = Resampling.min
    elif resampling == "max":
        resampling = Resampling.max
    else:
        print("Warning: Resampling method not recognized, switching to default 'nearest'")
        resampling = Resampling.nearest
    if type(src_band) is list:
        num_bands = len(src_band)
    else:
        num_bands = 1
    with rasterio.open(src_filename) as src:
        if src.nodata is None:
            nd_value = 0
        else:
            nd_value = src.nodata
        if num_bands == 1:
            dst_array = np.ones((dst_height, dst_width), dtype=src.meta["dtype"]) * nd_value
        else:
            dst_array = np.ones((num_bands, dst_height, dst_width), dtype=src.meta["dtype"]) * nd_value
        warp.reproject(rasterio.band(src, src_band), destination=dst_array, dst_transform=dst_transform,
                       dst_crs=CRS.from_dict(init=dst_crs_init), dst_resolution=dst_transform[0], resampling=resampling)
    if return_nodata:
        return dst_array, nd_value
    return dst_array
