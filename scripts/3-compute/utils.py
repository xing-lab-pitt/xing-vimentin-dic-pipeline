import csv
import nd2reader
from nd2reader import ND2Reader
from pims import ImageSequenceND
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.transform
import scipy.stats
import PIL
from skimage.restoration import denoise_wavelet, estimate_sigma, denoise_nl_means, denoise_bilateral
import png
import gc
import shutil
import random
import matplotlib.patches as mpatches
from skimage.measure import regionprops
import pandas as pd
import math
from skimage.io import imread

mask_cast_type = ">i4"
mask_mode = "L"

# location of processed info
# center info will be saved to figure_dir
center_filename = "centers_data.csv"
crop_filename = "crop_data.csv"
# figure_dir = './figures/test_max_projection_only'
# crop_dir = './figures/crops_max_projection_info_with_masks'
figure_dir = "./figures/test_crop_new"
crop_dir = "./figures/crops_info_with_masks"
mask_folder_name = "masks"  # name of mask folder in crop_dir

# if we only want to generate crop but not to find all signal candidates
gen_crop_only = False
# if only use max projection for finding stage 1 centers
process_max_projection_only = True

bounding_size = 10  # for averaging each pixel values. can be disregarded
collapse_dist_threshold = 10  # minimum distance between centers
collapse_dist_threshold_3d = 15
# whether generate plot for each plot for debug. False during production phase
save_fig = True
max_center_num_per_2d_image = 7  # how many centers per crop?
# max distance of two points with same track id in consecutive time frame
tracking_dist_threshold = 20
tracking_gap = 15  # max gaps allowed when map signals to previous time point signals
# how many points to consider as center candidates according to Gaussian
# distribution?
sampling_quantile_per_image = 0.9
# for processing simple background information of a center
background_boundings = [30, 30, 2]
# filtering centers: how many standard deviations should each center pixel
# value be larger than background average? (excluding signal area)
filter_contrast_factors = np.linspace(0, 5, 5)
min_traj_len = 7  # minimum trajectory to be considered as a true signal
traj_circle_radius = 5  # radius of point in trajectory

cell_radius = 200  # cell radius used in filtering signals
max_allowed_signals_per_cell = 3


def show_image(image):
    # plt.close()
    plt.imshow(image)
    return image


def show_images(images):
    plt.close()
    fig, axes = plt.subplots(len(images))
    for i, image in enumerate(images):
        axes[i].imshow(image)
    plt.show()


def read_nd2_specific(path, c, t, z):
    images = ND2Reader(path)
    image = images.get_frame_2D(c=c, t=t, z=z)
    return image


def read_nd2_along_z(path, c, t):
    """
    :param path:
    :param c:
    :param t:
    :return: x x y x z
    """
    images = ND2Reader(path)
    channel_num = len(images.metadata["channels"])
    z_levels = images.metadata["z_levels"]
    count = 0
    total = len(z_levels)
    res_images = []
    for z in z_levels:
        image = images.get_frame_2D(c=c, t=t, z=z)
        res_images.append(image)
        count += 1
        # print('%d/%d images loaded' % (count, total))

    res = np.array(res_images)
    res = np.moveaxis(res, 0, -1)
    return res


def read_nd2(path, z_limit=None, t_limit=None):
    """
    returns images: T x Z x C x X x Y
    """
    images = ND2Reader(path)
    print("pixel type:", images.pixel_type)
    # for key in images.metadata:
    #     print(key, images.metadata[key])
    print(images.sizes)
    z_levels = images.metadata["z_levels"]
    frames = images.metadata["frames"]
    channel_num = len(images.metadata["channels"])
    count = 0
    data = []
    total = len(z_levels) * len(frames) * channel_num
    t_count = 0
    for t in frames:
        if t_limit and t_count >= t_limit:
            break
        data.append([])
        t_count += 1
        z_count = 0
        for z in z_levels:
            if z_limit and z_count >= z_limit:
                break
            z_count += 1
            data[t].append([])
            for c in range(channel_num):
                image = images.get_frame_2D(c=c, t=t, z=z)
                # print(image.shape)
                data[t][z].append(image)
                count += 1
                if count % 10 == 0:
                    print("%d/%d images loaded" % (count, total))
    print(count, "images loaded")
    res = np.array(data)
    return res


def make_dir(path, abort=True):
    """
    Discription - Create a folder for given path.
    Input       - (String) path.
    """
    if os.path.exists(path):
        print(path + " : exists")
        if abort:
            exit(0)
        elif os.path.isdir(path):
            print(path + " : is a directory, continue using the old one")
            return False
        else:
            print(path + " : is not a directory, creating one")
            os.makedirs(path)
            return True
    else:
        os.makedirs(path)
        return True


def load_images_from_dirs(dirs, ext="tif", limit=float("inf"), limit_per_dir=float("inf"), return_filename=False):
    """
    Description - Read all files with desired extention in a folder list
    Input       - (list of string) list of folder res_paths
                - (string) extention of the file_ext
                - (float) number of files to read
                - (bool) if Ture, the file names will be returned
    Output      - (list of ndarrays) an list of image ndarrays.
                - (list of string) if return_filename=True, then a list of file names will be returned.

    """
    if not limit:
        limit = float("inf")
    if not limit_per_dir:
        limit_per_dir = float("inf")
    files = []
    filenames = []
    count = 0
    total = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                total += 1  # total - total number of the desired file in the folders
    for d in dirs:
        count_per_dir = 0
        for f in os.listdir(d):
            if f.endswith(ext):
                try:
                    image = skimage.io.imread(os.path.join(d, f))
                except Exception as e:
                    print(e)
                filenames.append(f)
                files.append(image)
                count += 1
                count_per_dir += 1
                print("reading %d/%d images" % (count, total))
                if count >= limit:
                    if return_filename:
                        return files, filenames
                    else:
                        return files
                if count_per_dir >= limit_per_dir:
                    print("skip remaining images in %s" % d)
                    break

    if return_filename:
        return files, filenames
    else:
        return files


def load_nd2_path_from_dirs(dirs, ext="nd2", limit=float("inf")):
    res_paths = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                path = os.path.join(d, f)
                res_paths.append(path)
                count += 1
                if count >= limit:
                    return res_paths
    return res_paths


# def show_image_distribution(image):


def normalize(vec):
    pass


def print_np_vec(v):
    for i in range(len(v)):
        for j in range(len(v[i])):
            print(v[i][j], end=", ")
        print()


def plot_contours(ax, contours):
    for contour in contours:
        ax.plot(contour)


def plot_distribution(ax, image, q=0.99):
    data = image.reshape(-1)
    var = scipy.stats.norm(loc=np.mean(data), scale=np.std(data))
    # print(var.ppf(q))
    threshold = var.ppf(q)
    if threshold:
        data = data[np.where(data > threshold)]
    ax.hist(data, bins=200)


def count_image_gaussian_quantile(image, q=0.99):
    assert len(image.shape) == 2
    data = image.reshape(-1)
    var = scipy.stats.norm(loc=np.mean(data), scale=np.std(data))
    threshold = var.ppf(q)
    size = np.sum(data > threshold)
    return size


stats_header = [
    "center intensity",
    "group size",
    "background average",
    "background pixel number",
    "background_bounding_x",
    "background_bounding_y",
    "background_bounding_z",
    "box_x_min",
    "box_x_max",
    "box_y_min",
    "box_y_max",
    "box_z_min",
    "box_z_max",
    "crop_id",
]
CENTER_INTENSITY_IND = 0
BACKGROUND_AVERAGE = 2
CROP_STAT_IND = 13
X_BOX_IND = 7
Y_BOX_IND = 9
Z_BOX_IND = 11
BC_ZSCORE_IND = 14


def get_bound_from_stats(stats):
    return stats[X_BOX_IND : Z_BOX_IND + 1]


def save_centers_data(data, path):
    """
    data: centers_num x 2
    each entry represents a result from an image
    centers contain a list of centers
    stats contain a list of stats
    """
    with open(path, "w+", newline="") as f:
        writer = csv.writer(f)
        header = ["t", "z", "c", "x", "y"] + stats_header
        writer.writerow(header)
        for i in range(len(data[0])):
            center = data[0][i]
            stat = data[1][i]
            (
                t,
                z,
                c,
                center_intensity,
                group_size,
                background_avg,
                background_size,
                background_bounding_x,
                background_bounding_y,
                background_bounding_z,
            ) = stat[:10]
            bd = stat[10:16]
            crop_id = stat[16]
            row = (
                [
                    t,
                    z,
                    c,
                    center[0],
                    center[1],
                    center_intensity,
                    group_size,
                    background_avg,
                    background_size,
                    background_bounding_x,
                    background_bounding_y,
                    background_bounding_z,
                ]
                + bd
                + [crop_id]
            )
            writer.writerow(row)


def read_centers_data(path):
    """
    return: two dicts with key (t, z, c),
    containing centers, stats data at t, z, c respectively.
    """
    res = []
    t_centers = []
    t_stats = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        centers = {}
        stats = {}
        is_header = True
        header = None
        for row in reader:
            if is_header:
                header = row
                is_header = False
                continue
            # print(row)
            row = [float(x) for x in row]
            t, z, c = row[0:3]
            t, z, c = int(t), int(z), int(c)
            center = row[3:5]
            stat = row[5:]

            if not (t, z, c) in centers:
                centers[t, z, c] = []
                stats[t, z, c] = []
            centers[t, z, c].append(center)
            stats[t, z, c].append(stat)
        data = tuple([centers, stats])
        return data


active_circles = []


def plot_traj_maps(ax, image, maps, draw_traj=False):
    """
    given a list of mapping {centerId:pos}
    draw trajectories
    """
    global active_circles
    all_keys = []
    for mapping in maps:
        all_keys.extend(list(mapping.keys()))
    all_keys = set(all_keys)

    # corner case: no tracked filtered points
    if len(all_keys) == 0:
        return

    global active_circles
    for circle in active_circles:
        circle.set_visible(False)
        circle.remove()

    active_circles = []
    max_key = max(all_keys) + 1
    for key in all_keys:
        traj = []
        for t in range(len(maps)):
            if key in maps[t]:
                traj.append(maps[t][key])
        # draw trajectories
        x = [center[1] for center in traj]
        y = [center[0] for center in traj]
        color = [(key + 1) / max_key] * 3 + [1]
        if draw_traj:
            ax.plot(x, y, c=color)
        # ax.scatter(x, y, s=80, facecolors='none', edgecolors=color)
        # ax = plt.gca()

        if key in maps[-1]:
            cx, cy, cz = traj[-1]
            circle = plt.Circle((cy, cx), traj_circle_radius, color="r", fill=False)
            active_circles.append(circle)
            ax.add_artist(circle)


def write_comp_data(data, path, kernel):
    with open(path, "w+") as f:
        f.write(str(len(data)) + " " + str(len(data[0])) + " " + str(kernel) + "\n")
        for vec in data:
            for num in vec:
                f.write("%d " % num)
            f.write("\n")


def read_comp_data(path):
    with open(path, "r") as f:
        res = []
        for row in f:
            row = row.strip().replace("\n", "").split(" ")
            row = [float(r) for r in row]
            res.append(row)
    return np.array(res)


def get_max_centers_series(t_centers_data, channel):
    """
    t_centers_data[t][1]=max_centers
    """
    t_centers = []
    t_center2stats = []
    for data in t_centers_data:
        data = data[channel]
        max_centers = data[0]
        center2stats = data[1]
        t_centers.append(max_centers)
        t_center2stats.append(center2stats)
    return t_centers, t_center2stats


def get_center_background(images, center, group, boundings):
    """
    boundings: x, y, z boundings
    group: 2D group
    caution: images are in z,x,y order due to nd2, center and center are in x y z order
    """
    center = [int(x) for x in center]
    shape = images.shape
    group = set([tuple(int(x) for x in group_center) for group_center in group])
    # print('debug get center background: group size=', len(group), list(group)[0])
    acc = 0
    count = 0
    for x in range(center[0] - boundings[0], center[0] + boundings[0]):
        for y in range(center[1] - boundings[1], center[1] + boundings[1]):
            for z in range(center[2] - boundings[2], center[2] + boundings[2]):
                if x < 0 or y < 0 or z < 0 or x >= shape[1] or y >= shape[2] or z >= shape[0] or (x, y) in group:
                    continue
                else:
                    acc += images[z, x, y]
                    count += 1
    if count == 0:
        return 0, count
    return acc / count, count


def get_centers_background(image, centers, groups, boundings=[100, 100, 2]):
    res = []
    for i in range(len(centers)):
        center = centers[i]
        group = groups[i]
        background = get_center_background(image, center, group, boundings)
        res.append(background)
    return res


def get_center_stats(image, z_images, centers, groups, background_boundings=[10, 10, 2]):
    """
    return #centers x stats
    stats: [[center intensity, group length, ...]]
    """
    stats = []

    for j in range(len(centers)):
        center = centers[j]
        group = groups[j]
        center_intensity = image[center[0], center[1]]
        background_avg, background_size = get_center_background(z_images, center, group, background_boundings)
        stat = [
            center_intensity,
            len(group),
            background_avg,
            background_size,
            background_boundings[0],
            background_boundings[1],
            background_boundings[2],
        ]
        stats.append(stat)
        # print('debug stat:', stat)
    return stats


def get_2d_center_bg_avg(image, center, boundings, signal_size=7):
    if signal_size >= boundings[0] or signal_size >= boundings[1]:
        assert False, "cell size larger than boundings"
    x1, x2 = center[0] - boundings[0], center[0] + boundings[0]
    y1, y2 = center[1] - boundings[1], center[1] + boundings[1]
    sub = image[x1:x2, y1:y2]
    ids = np.full(sub.shape, True)
    ids[
        boundings[0] - signal_size : boundings[0] + signal_size, boundings[1] - signal_size : boundings[1] + signal_size
    ] = False
    sub = sub[ids]
    avg = np.mean(sub.flatten())
    std = np.std(sub.flatten())
    return avg, std


def get_2d_center_avg(image, center, boundings):
    x1, x2 = center[0] - boundings[0], center[0] + boundings[0]
    y1, y2 = center[1] - boundings[1], center[1] + boundings[1]
    avg = np.mean(image[x1:x2, y1:y2].flatten())
    std = np.std(image[x1:x2, y1:y2].flatten())
    return avg, std


track_header = (
    ["t", "c", "x", "y", "z", "track_id"] + stats_header + ["bc_score", "bc_center_intensity", "bc_bg_avg", "bc_bg_std"]
)


def gen_tracking_datarows(mappings_list, t_center2stats, channel):
    assert len(mappings_list) == len(t_center2stats)
    rows = []
    for t in range(len(mappings_list)):
        mapping = mappings_list[t]
        center2stats = t_center2stats[t]
        for track_id in mapping:
            center = mapping[track_id]
            stats = center2stats[tuple(center)]
            row = [t, channel] + list(center) + [track_id] + list(stats)
            rows.append(row)
    # print('debug gen data rows, results...')
    # for row in rows:
    #     print(row)
    return rows


def gen_tracking_datarows_from_mapping_list_only(mappings_list, channel, t_track2cell=None):
    rows = []
    for t in range(len(mappings_list)):
        mapping = mappings_list[t]
        if not (t_track2cell is None):
            track2cell = t_track2cell[t]
        else:
            track2cell = None
        for track_id in mapping:
            center = mapping[track_id]
            row = [t, center[2], channel] + list(center[:2]) + [track_id]
            if not (track2cell is None):
                cell = track2cell[track_id]
                if cell is None:
                    row += [None] * 6
                else:
                    row += [cell.id, cell.time] + list(cell.regionprop.bbox[:4])
            rows.append(row)
    return rows


def save_tracking_data(rows, path, track_header=track_header):
    with open(path, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(track_header)
        for row in rows:
            writer.writerow(row)


def read_tracking_data(path):
    """
    :param path:
    :return: two dicts: tc2centers, tc2stats
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        centers = {}
        stats = {}
        is_header = True
        header = None
        for row in reader:
            if is_header:
                header = row
                is_header = False
                continue
            # print(row)
            row = [float(x) for x in row]
            t, z, c = row[0:3]
            t, z, c = int(t), int(z), int(c)
            center = row[3:5] + [z]
            stat = row[5:]

            if not (t, c) in centers:
                centers[t, c] = []
                stats[t, c] = []
            centers[t, c].append(center)
            stats[t, c].append(stat)
        data = tuple([centers, stats])
        return data


def tracking_data_to_signal_mapping(tc2centers, tc2stats, return_stats=False):
    """
    :param tc2centers:
    :param tc2stats:
    :return: dict, c => t => trackid => center
    """
    max_t, max_c = -1, -1
    for t, c in tc2centers:
        max_t, max_c = max(max_t, t), max(max_c, c)
    channel2mappings = {}
    channel2mappings_stats = {}
    for t, c in tc2centers:
        if not (c in channel2mappings):
            channel2mappings[c] = [{} for t in range(max_t + 1)]
            channel2mappings_stats[c] = [{} for t in range(max_t + 1)]
        centers = tc2centers[t, c]
        stats = tc2stats[t, c]
        for i in range(len(centers)):
            stat = stats[i]
            center = centers[i]
            track_id = stat[0]
            channel2mappings[c][t][track_id] = center
            channel2mappings_stats[c][t][track_id] = stats
    if return_stats:
        return channel2mappings, channel2mappings_stats
    return channel2mappings


def get_filename_from_path(path, extension=False):
    """
    returns filename from a path, without extension
    """
    basename = os.path.basename(path)
    res = os.path.splitext(basename)  # filename, ext
    if not extension:
        res = res[0]
    return res


def get_mask_path(mask_dir, cur_t, cur_c, cur_z):
    return os.path.join(mask_dir, "mask_T=%d_C=%d_Z=%d.png" % (cur_t, cur_c, cur_z))


crop_info_header = ["t", "c", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


def save_crop_info(bd_info, masks, crop_path, mask_dir, cur_t, cur_c):
    """
    masks should be 3d
    """
    with open(crop_path, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(crop_info_header)
        for t, c in bd_info:
            for bd in bd_info[t, c]:
                row = [t, c] + list(bd)
                writer.writerow(row)
    if not (masks is None):
        # check mask shape
        assert len(masks.shape) == 3
        for z in range(0, masks.shape[2]):
            mask = masks[:, :, z]
            mask_path = get_mask_path(mask_dir, cur_t, cur_c, z)
            converted_mask = mask.astype(mask_cast_type, casting="unsafe")
            # im = PIL.Image.fromarray(converted_mask, mode=mask_mode)
            # im.save(mask_path)
            writer = png.Writer(mask.shape[1], mask.shape[0], bitdepth=16)
            with open(mask_path, "wb+") as f:
                writer.write(f, converted_mask)


def read_crop_info(crop_path):
    bd_info = {}
    t_max, c_max = -1, -1
    with open(crop_path, "r") as f:
        reader = csv.reader(f)
        is_header = True
        header = None
        for row in reader:
            if is_header:
                is_header = False
                header = row
                continue
            t, c = int(row[0]), int(row[1])
            t_max, c_max = max(t_max, t), max(c_max, c)
            bd = [int(x) for x in row[2:]]
            if (t, c) in bd_info:
                bd_info[t, c].append(bd)
            else:
                bd_info[t, c] = [bd]

    for t in range(0, t_max + 1):
        for c in range(0, c_max + 1):
            if not ((t, c) in bd_info):
                bd_info[t, c] = []
    # print('debubg bd_info:', bd_info)
    return bd_info


def read_mask(path):
    mask = np.array(PIL.Image.open(path, mode="r"))
    return mask


def read_masks(mask_dir, t, c, z_levels):
    """
    return a 3d mask: X x Y x Z
    """
    masks = []
    for z in range(z_levels):
        path = get_mask_path(mask_dir, t, c, z)
        if not os.path.exists(path):
            return None
        mask = np.array(PIL.Image.open(path, mode="r"))
        # plt.imshow(mask)
        # plt.show()
        masks.append(mask)
    masks = np.array(masks)
    masks = np.moveaxis(masks, 0, 2)
    return masks


def read_all_masks(mask_dir, at, ac, az):
    """
    input: path, #t, #c, #z
    output: masks: t x Z x c x X x Y
    """
    all_masks = []
    for t in range(at):
        all_masks.append([])
        for c in range(ac):
            all_masks[t].append(read_masks(mask_dir, t, c, az))
    print("debug shape:", np.array(all_masks))
    return np.moveaxis(np.array(all_masks), 4, 1)


def save_png_image(image, path, bitdepth=16):
    """
    save png image with bit depth 16
    """
    writer = png.Writer(image.shape[1], image.shape[0], bitdepth=bitdepth)
    with open(path, "wb+") as f:
        writer.write(f, image)


def save_tiff_image(image, path, bitdepth=64, mode="RGB"):
    with open(path, "wb+") as f:
        im = PIL.Image.fromarray(image.astype(np.uint8), mode=mode)
        im.save(path)


def convert_nd2_to_png(nd2_path, dest):
    make_dir(dest, abort=False)
    # images = read_nd2(nd2_path)
    images = ND2Reader(nd2_path)
    # t_levels = images.shape[0]
    # z_levels = images.shape[1]
    t_levels = images.metadata["frames"]
    z_levels = images.metadata["z_levels"]
    channels = images.metadata["channels"]
    channel_num = len(images.metadata["channels"])
    # channel_num = images.shape[2]
    total = channel_num * len(t_levels) * len(z_levels)
    nd2_name = get_filename_from_path(nd2_path)

    # check dtype < 16bits
    print("image dtype:", read_nd2_specific(nd2_path, 0, 0, 0).dtype)
    assert read_nd2_specific(nd2_path, 0, 0, 0).dtype == np.uint16
    count = 0
    for c in range(channel_num):
        for t in t_levels:
            for z in z_levels:
                # print(t, z, c)
                # filename = nd2_name + '_T=' + str(t) + '_C=' + str(c) + '_Z=' + str(z) + '.png'
                filename = nd2_name + "_T=" + str(t) + "_C=" + str(c) + "_Z=" + str(z) + ".tiff"
                path = os.path.join(dest, filename)
                count += 1
                if os.path.exists(path):
                    print("skipping", count)
                    continue
                if count % (10) == 0:
                    print("%d/%d converted" % (count, total), flush=True)
                    gc.collect()
                # image = images[t, z, c, ...]
                try:
                    image = images.get_frame_2D(c=c, t=t, z=z)
                    # save_png_image(image, path)
                    save_tiff_image(image, path)
                    pass
                except Exception as e:
                    print("error")
                    print(e)


def plot_boxes_on_image(ax, bds):
    """
    bds: list of [xmin, xmax, ymin, ymax, ...]
    """
    for bd in bds:
        rec = plt.Rectangle(
            (
                bd[2],
                bd[0],
            ),
            bd[3] - bd[2],
            bd[1] - bd[0],
            color="r",
            fill=False,
        )
        active_circles.append(rec)
        ax.add_artist(rec)


def get_nikon_position_from_nd2_converted_filename(filename):
    idx = filename.find("_XY")
    idx = idx + 1
    temp = idx
    while filename[temp] != "_" and filename[temp] != ".":
        temp += 1
    end = temp
    pos = filename[idx + 2 : end]
    pos = pos.lstrip("0")
    return pos


def get_nikon_nd2_convention_tzc_from_name(filename):
    """
    input: a filename with nikon conversion convention
    output: t, z, c coords extracted

    Note: all t, z, c should START from 1
    example:
    input: 293t_xy07_z15_mono_t01.tif
    output: 1, 15, 1
    input: 293t_xy07_z15_mono1_t01.tif
    output: 1, 15, 2
    """

    def parse(name, pre):

        # not sensitive to upper/lower cases
        name = name.lower()
        pre = pre.lower()

        idx = name.find(pre)
        if idx == -1:
            """
            need to check this case.
            Default bahvior of nikon conversion is when there is only 1 Z or 1 time point, no T or Z in filename
            """
            # assert False, pre + ' not found in ' + name
            return 1
        else:
            temp = idx + 1
            while filename[temp] != "_" and filename[temp] != ".":
                temp += 1
            end = temp
            res = name[idx + len(pre) : end]
            res = res.lstrip("0")
            return res

    t = int(parse(filename, "_T"))
    z = int(parse(filename, "_Z"))
    c = parse(filename, "Mono")
    if c == "":
        c = 1
    else:
        c = int(c) + 1
    # print(filename)
    # print('t=', t, 'z=', z, 'c=', c)
    return t, z, c


def move_pos_img_to_subdirs(image_dir, ext="png"):
    paths = load_nd2_path_from_dirs([image_dir], ext=ext, limit=float("inf"))
    # print(paths)
    for path in paths:
        filename, file_ext = get_filename_from_path(path, extension=True)
        pos = get_nikon_position_from_nd2_converted_filename(filename + file_ext)
        t, z, c = get_nikon_nd2_convention_tzc_from_name(filename + file_ext)
        subdir = os.path.join(image_dir, "channel" + str(c), pos)
        if not os.path.exists(subdir):
            make_dir(subdir, abort=False)
        new_path = os.path.join(subdir, filename + file_ext)
        shutil.move(path, new_path)
        # print('filename, ext:', filename, ext)
        print("move %s to %s" % (path, new_path))


def read_2d_png_exp_data(path):
    dirs = [path]
    images, filenames = load_images_from_dirs(dirs, ext="tif", return_filename=True)
    tzc_list = []
    # only that converted nd2 count starting from 1
    # so no need to add 1 to shape later
    t_max, z_max, c_max = 0, 0, 0
    for i in range(len(images)):
        image = images[i]
        filename = filenames[i]
        t, z, c = get_nikon_nd2_convention_tzc_from_name(filename)
        tzc_list.append([t, z, c])
        t_max, z_max, c_max = max(t_max, t), max(z_max, z), max(c_max, c)

    data = np.zeros((t_max, z_max, c_max, images[0].shape[0], images[0].shape[1]), dtype=images[0].dtype)
    for i in range(len(images)):
        image = images[i]
        t, z, c = tzc_list[i]
        # need starting from 0 indices here
        data[t - 1, z - 1, c - 1, ...] = np.array(images[i])

    return data


def read_2d_classified_data(path, limit=float("inf")):  # , num_class=3):
    dirs = [path]
    images, filenames = load_images_from_dirs(dirs, ext="tif", return_filename=True, limit=limit)
    # tzc_list = []
    tzc2image = {}
    # only that converted nd2 count starting from 1
    # so no need to add 1 to shape later
    # random.shuffle(images)
    for i in range(min(limit, len(images))):
        image = images[i]
        filename = filenames[i]
        t, z, c = get_nikon_nd2_convention_tzc_from_name(filename)
        # print("check extraction results", filename, t, z, c)
        assert not (t - 1, z - 1, c - 1) in tzc2image, "duplicate prob map, check parsing or source dir"
        # note index starts from 0, so -1
        tzc2image[
            (
                t - 1,
                z - 1,
                c - 1,
            )
        ] = image
    return tzc2image


def read_2d_classified_data_np_arr(path, limit=float("inf")):  # , num_class=3):
    dirs = [path]
    images, filenames = load_images_from_dirs(dirs, ext="tif", return_filename=True, limit=limit)
    # tzc_list = []
    tzc2image = {}
    # only that converted nd2 count starting from 1
    # so no need to add 1 to shape later

    # random.shuffle(images)
    num_t, num_z, num_c = 0, 0, 0
    for i in range(min(limit, len(images))):
        image = images[i]
        filename = filenames[i]
        t, z, c = get_nikon_nd2_convention_tzc_from_name(filename)
        num_t = max(t, num_t)
        num_z = max(z, num_z)
        num_c = max(c, num_c)

    image_dim = images[0].shape
    res_data = np.zeros([num_t, num_z, num_c] + list(image_dim))
    for i in range(min(limit, len(images))):
        image = images[i]
        filename = filenames[i]
        t, z, c = get_nikon_nd2_convention_tzc_from_name(filename)
        res_data[t - 1, z - 1, c - 1] = image
        # # print("check extraction results", filename, t, z, c)
        # assert not (t-1, z-1, c-1) in tzc2image, "duplicate prob map, check parsing or source dir"
        # note index starts from 0, so -1
        # tzc2image[(t-1, z-1, c-1,)] = image
    return res_data


def read_2d_classified_data_specific_tzc(dir_path, t, z, c, ext="tif"):
    paths = load_nd2_path_from_dirs([dir_path], ext=ext)
    filenames = ["".join(get_filename_from_path(path, extension=True)) for path in paths]
    for i, filename in enumerate(filenames):
        path = paths[i]
        ft, fz, fc = get_nikon_nd2_convention_tzc_from_name(filename)
        print(ft, fz, fc)
        if ft - 1 == t and fz - 1 == z and fc - 1 == c:
            image = skimage.io.imread(path)
            return image


def get_data_pair(tzcxy_images, tzc2image):
    X, Y = [], []
    for t, z, c in tzc2image:
        X.append(tzcxy_images[t, z, c, ...])
        Y.append(
            tzc2image[
                (
                    t,
                    z,
                    c,
                )
            ]
        )
    return X, Y


def assemble_prob_map(images, offsets, height, width, sh, sw, n_channel):
    res = np.zeros((height, width, n_channel), dtype=np.float)
    for i in range(len(images)):
        image = images[i]
        offset = offsets[i]
        origin = res[offset[0] : offset[0] + sh, offset[1] : offset[1] + sw, ...]
        res[offset[0] : offset[0] + sh, offset[1] : offset[1] + sw, ...] = np.maximum(image, origin)
    return res


def probmap2labels(probmap):
    n_class = probmap.shape[-1]
    other_total = np.prod(np.array(probmap.shape[:-1]))
    ids = np.argmax(probmap, axis=len(probmap.shape) - 1)
    ids = np.array(ids)
    res = np.zeros(probmap.shape)
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            res[i, j, ids[i, j]] = 1
    return res.reshape(probmap.shape)


def tzcDict2npArray(tzc2some):
    max_t, max_z, max_c = -1, -1, -1
    shape = None
    for t, z, c in tzc2some:
        max_t, max_z, max_c = max(max_t, t), max(max_z, z), max(max_c, c)
        shape = tzc2some[t, z, c].shape

    res = np.zeros((max_t + 1, max_z + 1, max_c + 1, shape[0], shape[1], shape[2]))
    for t in range(max_t + 1):
        # res.append([])
        for z in range(max_z + 1):
            # res[t].append([])
            for c in range(max_c + 1):
                res[t, z, c, ...] = tzc2some[t, z, c]
    # res = np.array(res, copy=False)
    return res


def save_max_projected_images(images, dest, channel_num=2, t_start=0):
    prefix_name = "max_projected"

    def gen_name(t, c):
        return prefix_name + "_T=" + t + "_C=" + c

    for t in range(t_start):
        for c in range(channel_num):
            mask = images[t, c, ...]
            writer = png.Writer(mask.shape[1], mask.shape[0], bitdepth=16)
            image_path = os.path.join(dest, gen_name(t, c))
            with open(image_path, "wb+") as f:
                writer.write(f, mask)


def bb_intersection_over_union(boxA, boxB):
    """
    source: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    :return:
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def distance(c1, c2):
    # return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** .5
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.linalg.norm(c1 - c2)


def print_mappings(mappings):
    print("mappings result:")
    for i, mapping in enumerate(mappings):
        print("at %d, mappings:" % i, mappings[i])


def save_mask(mask, mask_path):
    # converted_mask = mask.astype('i2', casting='unsafe')
    writer = png.Writer(mask.shape[1], mask.shape[0], bitdepth=16)
    with open(mask_path, "wb+") as f:
        writer.write(f, mask)


def get_bds_from_bboxes(bboxes):
    assert len(bboxes[0]) % 2 == 0
    res = []
    dims = len(bboxes[0]) // 2
    for bbox in bboxes:
        bd = []
        for i in range(dims):
            # note bbox is [min, max) but bd is [min, max]
            bd += [bbox[i], bbox[dims + i] - 1]
        res.append(bd)
    return res


def convert_multi_tif_to_single_tifs():
    """
    TODO: not needed for now
    Fiji script has solved this problem
    """
    assert False, "Ke: Not implemented :("
    path = "/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse/bg_corrected_data/Corrected_TRITC_data.tif"
    multi_tif_images, multi_tif_filenames = load_images_from_dirs(paths, ext="tif", return_filename=True, limit=None)
