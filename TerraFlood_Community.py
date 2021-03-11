import os
import sys
import numpy as np
import geopandas as gpd
from datetime import datetime as dt
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
from owslib.wms import WebMapService as WMS
from PIL import Image
from TFmodules import cataloguefunctions as caf
from TFmodules import tiledthresholdingfunctions as ttf
from TFmodules import terrafloodfunctions as tf

location = sys.argv[1]  # Flemish community
startdate_tag = sys.argv[2]  # YYYYMMDD
enddate_tag = sys.argv[3]  # YYYYMMDD
startdate = dt.strptime(startdate_tag, "%Y%m%d")
enddate = dt.strptime(enddate_tag, "%Y%m%d")

print("Creating flood maps for {} between {} and {}".format(location, dt.strftime(startdate, "%d-%m-%Y"),
                                                            dt.strftime(enddate, "%d-%m-%Y")))

wgs = "epsg:4326"
utm = "epsg:32631"
transformer_wgs_utm = Transformer.from_crs(wgs, utm, always_xy=True)
transformer_utm_wgs = Transformer.from_crs(utm, wgs, always_xy=True)
producttype_s1 = "urn:eop:VITO:CGS_S1_GRD_SIGMA0_L1"
s1_resolution = 10

dir_root = "/dir/to/data/folder/"
dir_data = os.path.join(dir_root, "Output")
dir_lc = "/dir/to/data/lc/"
dir_dem = "/dir/to/data/dem/"
dir_kbl = "/dir/to/data/kbl/"
df_communities = gpd.read_file("/dir/to/communities-shapefile/")\
    .to_crs(utm)

nd_dem = -9999
nd_pres = 99
nd_map = -1

ratio = 1.0
kernel_ws = 7
maxdist = 3
t_incvv = 3
t_incr = 3
mmu_holes = 5
mmu = 10
pw_sel = "bbk-risk"
pw_perc = 50
pf_sel = "risk"
ff_sel = "risk-depth"
fv_seeds = "of"
s1_min_coverage = 20

# ROI information
dir_location = os.path.join(dir_data, location)
if not os.path.exists(dir_location):
    os.mkdir(dir_location)
x_min, y_min, x_max, y_max = df_communities.loc[df_communities["NAAM"] == location, "geometry"].to_numpy()[0].bounds
x_min, y_min = np.floor([x_min, y_min])
x_max, y_max = np.ceil([x_max, y_max])
x_max += (s1_resolution - ((x_max - x_min) % s1_resolution))
y_max += (s1_resolution - ((y_max - y_min) % s1_resolution))
roi_extent = (x_min, y_min, x_max, y_max)
roi_extent_wgs = transformer_utm_wgs.transform(x_min, y_min) + transformer_utm_wgs.transform(x_max, y_max)
roi_shape = (int((x_max - x_min) / s1_resolution), int((y_max - y_min) / s1_resolution))  # width, height
roi_transform = rasterio.transform.from_bounds(*roi_extent, *roi_shape)
roi_metadata = {
    "crs": CRS.from_dict(init=utm),
    "transform": roi_transform,
    "width": roi_shape[0],
    "height": roi_shape[1]
}
pixel_resolution_x = roi_transform[0]
pixel_resolution_y = -roi_transform[4]

# Tree & PW presence data
print("{} - Gathering LC data...".format(dt.now()))
trees_filename = os.path.join(dir_location, "Trees.tif")
if not os.path.exists(trees_filename):
    trees = caf.mosaic_tiles(dir_lc, *roi_extent, roi_transform, *roi_shape, utm, "int8", nd_pres, query_string="Trees",
                             kbl_dir=dir_kbl, kbl_scale=0, resampling="max", fill_nodata=False,
                             mosaic_filename=trees_filename)
else:
    trees = caf.tif2ar(trees_filename)[0]
pw_filename = os.path.join(dir_location, "PW.tif")
if not os.path.exists(pw_filename):
    pw = caf.mosaic_tiles(dir_lc, *roi_extent, roi_transform, *roi_shape, utm, "int8", nd_pres, query_string="PW",
                          kbl_dir=dir_kbl, kbl_scale=0, resampling="max", fill_nodata=False,
                          mosaic_filename=pw_filename)
else:
    pw = caf.tif2ar(pw_filename)[0]

# DEM data
print("{} - Gathering DEM data...".format(dt.now()))
dem_filename = os.path.join(dir_location, "DEM.tif")
if not os.path.exists(dem_filename):
    dem = caf.mosaic_tiles(dir_dem, *roi_extent, roi_transform, *roi_shape, utm, "float32", nd_dem,
                           kbl_dir=dir_kbl, kbl_scale=8, resampling="average", fill_nodata=True,
                           mosaic_filename=dem_filename)
else:
    dem = caf.tif2ar(dem_filename)[0]

# OG data
print("{} - Gathering OG data...".format(dt.now()))
for label, layer in zip(["OGP", "OGF", "ODP_1000", "ODF_1000"],
                        ['Overstromingsgevaarkaarten-PLUVIAAL:overstroombaar_gebied_PLU_noCC',
                         'Overstromingsgevaarkaarten-FLUVIAAL:overstroombaar_gebied_FLU_noCC',
                         'Overstromingsgevaarkaarten-PLUVIAAL:waterdiepte_PLU_noCC_T1000',
                         'Overstromingsgevaarkaarten-FLUVIAAL:waterdiepte_FLU_noCC_T1000']):
    og_filename = os.path.join(dir_location, "{}.tif".format(label))
    if not os.path.exists(og_filename):
        wms_og = WMS("http://geoservice.waterinfo.be/OGRK/wms", timeout=120)
        img = wms_og.getmap(layers=[layer], srs=utm, bbox=roi_extent, size=roi_shape, format="image/GeoTiff")
        with open(og_filename, "wb") as out:
            out.write(img.read())
        with rasterio.open(og_filename) as src:
            og_profile = src.profile
            og_layer = src.read(1)
            if "OG" in label:
                og_layer = (og_layer == 1) * 3 + (og_layer == 2) * 2 + (og_layer == 3) * 1
            elif "OD" in label:
                og_layer = (og_layer == 1) * 13 + (og_layer == 2) * 38 + (og_layer == 3) * 75 + (og_layer == 4) * \
                           150 + (og_layer == 5) * 200
        with rasterio.open(og_filename, "w", **og_profile) as dst:
            dst.write(og_layer.astype(og_profile["dtype"]), indexes=1)
    else:
        og_layer = caf.tif2ar(og_filename)[0]
    if label == "OGP":
        ogp = og_layer
    elif label == "OGF":
        ogf = og_layer
    elif label == "ODP_1000":
        odp = og_layer / 100
    elif label == "ODF_1000":
        odf = og_layer / 100
    del og_layer

# S-1 data
products_s1 = caf.find_products(urn=producttype_s1, start=startdate.isoformat(), end=enddate.isoformat(),
                                lonmin=roi_extent_wgs[0], latmin=roi_extent_wgs[1], lonmax=roi_extent_wgs[2],
                                latmax=roi_extent_wgs[3])
products_s1 = [el for el in products_s1 if el != "no products"]
map_filenames = []
for p_s1f in products_s1:
    p_s1f_date = caf.time_to_local(dt.strptime(p_s1f["productDate"], "%Y-%m-%dT%H:%M:%S.%fZ"))
    p_s1f_datetag = dt.strftime(p_s1f_date, "%Y%m%d-%H%M")
    p_s1f_id = p_s1f["productTitle"][-9:-5]
    dir_output = os.path.join(dir_location, p_s1f_datetag[:4])
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    s1f_vv_filepath = [el["filepath"] for el in p_s1f["files"] if el["title"] == "VV"][0]
    w = caf.get_window(s1f_vv_filepath, *roi_extent_wgs, wgs)[0]
    data_coverage = caf.get_datacoverage(s1f_vv_filepath, window=w)
    print("{} - Product {}: coverage = {}%".format(dt.now(), p_s1f["productTitle"], data_coverage))
    if data_coverage >= s1_min_coverage:
        # S-1 image pair
        print("{} - Gathering S-1 image pair...".format(dt.now()))
        p_s1r = caf.find_s1_ref(dt.strptime(p_s1f["productDate"], "%Y-%m-%dT%H:%M:%S.%fZ"), *roi_extent_wgs,
                                p_s1f["relativeOrbit"], p_s1f["bbox"], min_coverage=s1_min_coverage)
        p_s1r_date = caf.time_to_local(dt.strptime(p_s1r["productDate"], "%Y-%m-%dT%H:%M:%S.%fZ"))
        p_s1r_datetag = dt.strftime(p_s1r_date, "%Y%m%d-%H%M")
        p_s1r_id = p_s1r["productTitle"][-9:-5]
        sar_datetag = "{}_{}_{}-{}".format(p_s1f_datetag, p_s1r_datetag, p_s1f_id, p_s1r_id)
        sar_filename = os.path.join(dir_output, "{}_{}_S1.tif".format(location, sar_datetag))
        if not os.path.exists(sar_filename):
            ref_vv = 10 * np.log10(caf.warp_to_transform([el["filepath"] for el in p_s1r["files"] if el["title"]
                                                          == "VV"][0], roi_transform, *roi_shape, utm))
            ref_vh = 10 * np.log10(caf.warp_to_transform([el["filepath"] for el in p_s1r["files"] if el["title"]
                                                          == "VH"][0], roi_transform, *roi_shape, utm))
            ref_angle = 0.0005 * caf.warp_to_transform([el["filepath"] for el in p_s1r["files"] if el["title"]
                                                        == "angle"][0], roi_transform, *roi_shape, utm) + 29
            flood_vv = 10 * np.log10(caf.warp_to_transform([el["filepath"] for el in p_s1f["files"] if el["title"]
                                                            == "VV"][0], roi_transform, *roi_shape, utm))
            flood_vh = 10 * np.log10(caf.warp_to_transform([el["filepath"] for el in p_s1f["files"] if el["title"]
                                                            == "VH"][0], roi_transform, *roi_shape, utm))
            flood_angle = 0.0005 * caf.warp_to_transform([el["filepath"] for el in p_s1f["files"] if el["title"]
                                                          == "angle"][0], roi_transform, *roi_shape, utm) + 29
            del ref_angle, flood_angle
            sar_bandnames = ["rVH", "rVV", "fVH", "fVV"]
            caf.ar2tif(np.array([ref_vh, ref_vv, flood_vh, flood_vv]), sar_filename, utm, roi_transform,
                       dtype="float32", band_index=0, band_names=sar_bandnames)
        else:
            ref_vh, ref_vv, flood_vh, flood_vv = caf.tif2ar(sar_filename)[0]
        map_filename = "{}.png".format(sar_filename[:-4])
        if not os.path.exists(map_filename):
            tf.make_map_sar(np.array([ref_vh, ref_vv, flood_vh, flood_vv]), roi_extent, map_filename,
                            location, p_s1f_date)

        # S-1 band combinations
        print("{} - Calculating SAR band combinations...".format(dt.now()))
        inc_vv = flood_vv - ref_vv
        ratio_ref = ref_vv - ref_vh
        ratio_flood = flood_vv - flood_vh
        inc_r = ratio_flood - ratio_ref
        del ratio_flood, ratio_ref

        # Threshold selection
        thresh_file = os.path.join(dir_data, "Thresholds", "Thresholds_{}-{}.csv".format(p_s1f_datetag, p_s1f_id))
        if not os.path.exists(thresh_file):
            print("{} - Threshold selection...".format(dt.now()))
            with rasterio.open([el["filepath"] for el in p_s1f["files"] if el["title"] == "VV"][0]) as src:
                im = 10 * np.log10(src.read(1).astype("float32"))
            t_vv_ki, t_vv_o = ttf.tiled_thresholding(im, selection='Martinis', t_method=['KI', "Otsu"],
                                                     tile_dim=[200, 200], n_final=5, incomplete_tile_warning=False)[0]
            del im
            with rasterio.open([el["filepath"] for el in p_s1f["files"] if el["title"] == "VH"][0]) as src:
                im = 10 * np.log10(src.read(1).astype("float32"))
            t_vh_ki, t_vh_o = ttf.tiled_thresholding(im, selection='Martinis', t_method=['KI', "Otsu"],
                                                     tile_dim=[200, 200], n_final=5, incomplete_tile_warning=False)[0]
            del im
            with open(thresh_file, "w") as handle:
                handle.writelines(["{:.4f}, {:.4f} \n".format(t_vv_ki, t_vv_o),
                                   "{:.4f}, {:.4f} \n".format(t_vh_ki, t_vh_o)])
            t_vv = t_vv_ki
            t_vh = t_vh_ki
        else:
            print("{} - Loading thresholds...".format(dt.now()))
            with open(thresh_file, "r") as handle:
                lines = handle.readlines()
                t_vv = float(lines[0][:lines[0].index(",")])
                t_vh = float(lines[1][:lines[1].index(",")])
            del lines

        # TerraFlood maps
        tfco_filename = os.path.join(dir_output, "{}_{}_TerraFloodMap.tif".format(location, sar_datetag))
        if not os.path.exists(tfco_filename):
            # TerraFloodOb
            print("{} - Applying TerraFloodOb...".format(dt.now()))
            tf_ob = tf.apply_terraflood(flood_vv, flood_vh, ref_vv, ref_vh, inc_vv, inc_r, trees, pw, ogp, ogf, odp, odf,
                                        dem, t_vv, t_vh, t_incvv=3, t_incr=3, ratio=1.0, kernel_ws=7,
                                        maxdist=3, mmu=mmu, mmu_holes=mmu_holes, pf_selection=pf_sel, ff_inclusion=ff_sel,
                                        pw_selection=pw_sel, pw_perc=pw_perc, fv_seeds=fv_seeds)
            # TerraFloodPx
            print("{} - Applying TerraFloodPx...".format(dt.now()))
            tf_px = tf.apply_terrafloodpx(flood_vv, flood_vh, ref_vv, ref_vh, inc_vv, inc_r, trees, pw, ogp, ogf, odp, odf,
                                          dem, t_vv, t_vh, t_incvv=3, t_incr=3, mmu_core=mmu, mmu_holes=mmu_holes,
                                          pf_selection=pf_sel, ff_inclusion=ff_sel, pw_selection=pw_sel, pw_perc=pw_perc,
                                          fv_seeds=fv_seeds)
            # TerraFloodCombo
            print("{} - Applying TerraFloodCombo...".format(dt.now()))
            tf_combo = tf.combine_floodmaps(tf_ob, tf_px, pw, trees, pf_selection=pf_sel, ogp=ogp, ogf=ogf, mmu=mmu_holes,
                                            pw_perc=pw_perc)
            with rasterio.open(tfco_filename, "w", driver="GTiff", width=roi_shape[0], height=roi_shape[1], count=1,
                               crs=utm, transform=roi_transform, dtype="int16", nodata=nd_map) as dst:
                dst.write(tf_combo.astype("int16"), indexes=1)
        else:
            tf_combo = caf.tif2ar(tfco_filename)[0]
        map_filename = os.path.join(dir_output, "{}.png".format(tfco_filename[:-4]))
        map_filenames.append(map_filename)
        if not os.path.exists(map_filename):
            tf.make_floodmap(tf_combo, roi_extent, map_filename, location, p_s1f_date)

        print("------------------------------------------------------------------------------")

map_filenames.sort()
gif_filename = os.path.join(dir_output, "{}_{}-{}.gif".format(location, startdate_tag, enddate_tag))
if not os.path.exists(gif_filename):
    img, *imgs = [Image.open(f) for f in map_filenames]
    img.save(fp=gif_filename, format="GIF", append_images=imgs, save_all=True, duration=500, loop=0)
