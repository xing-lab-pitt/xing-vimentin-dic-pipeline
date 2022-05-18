import math
import os
import subprocess
import uuid

import cv2
import cv2 as cv
import matplotlib.cm
import numpy as np
import PIL
import scipy
import skimage
import skimage.segmentation
import sklearn
import utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from PIL import Image, ImageEnhance
from scipy import interpolate
from scipy import ndimage as ndi
from skimage import feature, measure
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.filters import threshold_local
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from utils import distance


def denoise_images(images, win_size=21):
    res = []
    o_shape = images.shape
    images = images.reshape([-1] + list(o_shape[-2:]))
    for image in images:
        # image = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)
        # image = utils.denoise_wavelet(image)
        # image = utils.denoise_nl_means(image)
        image = utils.denoise_bilateral(image, win_size=win_size)
        res.append(image)
    return np.array(res).reshape(o_shape)


def find_image_edges(images):
    res = []
    for image in images:
        not_sure_threshold = 2
        sure_threshold = 6
        # image = cv2.fastNlMeansDenoising(image,None, 20, 7, 21)
        # edges = cv2.Canny(image, not_sure_threshold, sure_threshold, 20)
        edges = feature.canny(image, low_threshold=not_sure_threshold, high_threshold=sure_threshold)
        edges[edges == 0] = np.ma.masked
        res.append(edges)
    return res


def collapse_centers(centers, groups, max_group_sample_size=None, dist_threshold=60, image=None):
    """
    collapse centers based on sorted (prior: high to low) list of center
    use brightest pixel as center group representative
    """
    cur_centers = list(centers)
    cur_groups = [list(group) for group in groups]

    while True:
        temp_centers = []
        temp_groups = []
        marked = [False for _ in range(len(cur_centers))]
        any_collapsed = False
        for i in range(0, len(cur_centers)):
            if marked[i]:
                continue
            center = cur_centers[i]
            group = cur_groups[i]
            collapsed = False
            for j in range(i + 1, len(cur_centers)):
                if marked[j]:
                    continue
                other = cur_centers[j]
                other_group = cur_groups[j]
                if distance(center, other) < dist_threshold:
                    temp_groups.append(group + other_group)
                    # based on biased group. prioritize brightest centers
                    # new_center = np.mean(temp_groups[-1][:max_group_sample_size],
                    #                      axis=0)
                    new_center = temp_groups[-1][0]  # use brightest point
                    temp_centers.append(new_center)
                    collapsed = True
                    any_collapsed = True
                    marked[j] = True
                    break

            if not collapsed:
                temp_centers.append(center)
                temp_groups.append(group)

        if not any_collapsed:
            break
        else:
            cur_centers = temp_centers
            cur_groups = temp_groups

    return cur_centers, cur_groups


def cluster_indices(pts, image, center_num=10, dist_threshold=60, max_group_sample_size=1000):
    # print('#selected pixels:', len(pts))
    origins = pts
    pts = list(pts)
    min_step = 200
    max_step = 100000
    i = 0
    z_tol = 1
    marked = [False for _ in range(len(pts))]
    marked_id = set()
    centers = []
    groups = []
    while i < len(pts) and center_num > len(centers):
        window = []
        window_vals = []
        while len(window) < min_step and i < len(pts):
            if not marked[i]:
                window.append(pts[i])
                window_vals.append(image[pts[i][0], pts[i][1]])
            i += 1
        std = np.std(window_vals, axis=0)
        mean = np.mean(window_vals, axis=0)

        while len(window) < max_step and i < len(pts):
            index = pts[i]
            val = image[index[0], index[1]]
            if std != 0 and (val - mean) / std < z_tol:
                i += 1
                window.append(index)
            else:
                break

        new_centers, new_groups = k_cluster_sklearn(window, k=center_num)

        # NOTE: order of + is important here: keep sorted
        centers = centers + new_centers
        groups = groups + new_groups
        centers, groups = collapse_centers(centers, groups, max_group_sample_size, dist_threshold=dist_threshold)

    return centers[:center_num], groups[:center_num]


def find_chrom_centers_simple_bound(
    image, bounding_size, max_center_num=10, sampling_quantile=0.97, collapse_dist_threshold=20
):
    shape = image.shape
    # print('calculating scores with bounding size=%d...' % bounding_size)
    # scores = find_bounding_sums_cpp(image, bounding_size)
    # scores = image
    scores = utils.denoise_bilateral(image)
    # utils.print_np_vec(scores)
    args = np.argsort(-scores, None)  # sort as flattened
    # C language (row major order)
    # choose top sampling_quantile coordinates regarding intensity,
    # then cluster indices based on these coordinates
    indices = (np.array(np.unravel_index(args, shape, order="C"))).T
    sample_size = utils.count_image_gaussian_quantile(image, sampling_quantile)
    centers, groups = cluster_indices(
        indices[:sample_size], image, center_num=max_center_num, dist_threshold=collapse_dist_threshold
    )
    return centers[:max_center_num], groups[:max_center_num]


def crop_by_sampling(image, sampling_quantile=0.99):
    # scores = utils.denoise_bilateral(image)
    scores = -image
    args = np.argsort(scores, None)
    # indices = (np.array(np.unravel_index(args, image.shape, order='C'))).T
    sample_size = utils.count_image_gaussian_quantile(image, q=sampling_quantile)
    # rows, cols = [x[0] for x in indices], [x[1] for x in indices]
    # rows, cols = rows[:sample_size], cols[:sample_size]
    mask = np.zeros(image.shape[0] * image.shape[1])
    mask[args[:sample_size]] = 1
    mask = mask.reshape(image.shape)
    return mask


def crop_by_local_threshold(image, block_size=301, offset=10):
    # image = utils.denoise_bilateral(image)
    threshold = threshold_local(image, block_size=block_size, offset=offset)
    mask = (image > threshold).astype(int)
    return mask


def bg_correction(image):
    # adapt code from weikang
    sample_step = 5
    I = image
    n_row, n_col = I.shape
    ctrl_x = []
    ctrl_y = []
    ctrl_z = []

    for i in np.arange(0, n_row, sample_step):
        for j in np.arange(0, n_col, sample_step):
            ctrl_x.append(i)
            ctrl_y.append(j)
            ctrl_z.append(I[i, j])
    ctrl_x = np.array(ctrl_x)
    ctrl_y = np.array(ctrl_y)
    ctrl_z = np.array(ctrl_z)

    nx, ny = I.shape[0], I.shape[1]
    lx = np.linspace(0, n_row, nx)
    ly = np.linspace(0, n_col, ny)

    # s value is important for smoothing
    tck = scipy.interpolate.bisplrep(ctrl_x, ctrl_y, ctrl_z, s=1e20)
    znew = scipy.interpolate.bisplev(lx, ly, tck)
    # func = scipy.interpolate.interp2d(ctrl_x, ctrl_y, ctrl_z, kind='quintic')
    # znew = func(lx, ly).T
    res = I - znew
    res[res < 0] = 0
    # plt.imshow(I - znew * 2)
    # fig, axes = plt.subplots(2,2)
    # axes[0, 0].imshow(I)
    # axes[0, 1].imshow(znew)
    # axes[1, 0].imshow(I-znew)
    # axes[1, 1].imshow(I-znew * 2)

    # axes[0, 0].set_title('original')
    # axes[0, 1].set_title('fitted surface')
    # axes[1, 0].set_title('image-surface')
    # axes[1, 1].set_title('image-surfaceXfactor')

    # plt.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # ax = fig.gca(projection='3d')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # x, y = np.meshgrid(lx, ly)
    # surf = ax.plot_surface(x, y, znew, cmap=matplotlib.cm.coolwarm,
    #                        linewidth=0)
    # ax.scatter(x, y, I - znew)
    # plt.show()
    return res


def bg_correction_whole_image(image, centers, stats):
    bds = set()
    image = np.copy(image)
    for stat in stats:
        xy_bds = tuple(stat[utils.X_BOX_IND : utils.X_BOX_IND + 4])
        xy_bds = tuple([int(x) for x in xy_bds])
        x1, x2, y1, y2 = xy_bds
        bds.add(xy_bds)

    for xy_bds in bds:
        x1, x2, y1, y2 = xy_bds
        try:
            image[x1:x2, y1:y2] = bg_correction(image[x1:x2, y1:y2])
        except Exception as e:
            print("bisplrep raises err, disregard...")
            print(e)
    return image


def find_contours(image, level=0.5):
    contours = measure.find_contours(image, level)
    return contours


def k_cluster_sklearn(pts, k=10, iters=3, min_group_size=None):
    """
    Note: pts are sorted based on image pixel values.
    return: a list of centers and a list of groups (list of list)
    """
    if len(pts) < k:
        # samples not enough
        return [], []
    # model = sklearn.cluster.KMeans(n_clusters=k)
    model = sklearn.mixture.GaussianMixture(n_components=k)
    cluster_ids = model.fit_predict(pts)
    groups = {}

    # note that group are in order of max -> min intensity
    for i in range(len(cluster_ids)):
        index = cluster_ids[i]
        if not (index in groups):
            groups[index] = []
        groups[index].append(pts[i])

    centers = []
    group_list = []
    for index in groups:
        group = groups[index]
        if min_group_size and len(group) < min_group_size:
            continue
        # center = np.mean(group, axis=0)
        center = group[0]  # use the highest val pixel as representative center
        centers.append(center)
        group_list.append(group)

    return centers, group_list


def find_3d_centers_in_frame_simple_z(image, data, dist_threshold):
    """
    image: Z x X x Y
    """
    centers = []
    center2stats = {}
    for z, c, z_centers, stats in data:
        d3_centers = [center + [z] for center in z_centers]
        d3_centers = [[int(x) for x in center] for center in d3_centers]
        centers += d3_centers
        for i in range(len(d3_centers)):
            center = d3_centers[i]
            center2stats[tuple(center)] = stats[i]
    centers = sorted(centers, key=lambda x: image[x[2], x[0], x[1]], reverse=True)
    # print('#centers in all z axis:', len(centers))
    groups = [[center] for center in centers]
    res_centers, groups = collapse_centers(centers, groups, dist_threshold=dist_threshold)
    max_centers = []
    avg_centers = []

    for i in range(len(groups)):
        group = groups[i]
        pixels = [image[c[2], c[0], c[1]] for c in group]
        # print('debug find 3D pts:', pixels, np.argmax(pixels))
        max_center = group[np.argmax(pixels)]
        # print('max center:', max_center, 'returned by collapse center:', res_centers[i])
        max_center = [int(x) for x in max_center]
        max_centers.append(max_center)
        assert int(distance(max_center, res_centers[i])) == 0
        avg_center = np.mean(group, axis=0)
        avg_centers.append(avg_center)

    # print('%d avg centers, %d max_centers' % (len(avg_centers), len(max_centers)))
    # stats = [center2stats[tuple(center)] for center in max_centers]
    return [max_centers, center2stats]


def find_3d_centers_in_frame_simple(image, data, dist_threshold):
    """
    input:
          data: a list of (z, c, centers, stats) found by each image
          image: z x c x x xy
    return: c x [avg_centers, max_centers, groups]
    """
    channels = image.shape[1]
    all_c = set([c for c in range(channels)])

    channel2centers = {}
    for c in all_c:
        c_data = []
        c_image = image[:, c, :, :]
        for z, cur_c, centers, stats in data:
            if cur_c != c:
                continue
            else:
                row = z, cur_c, centers, stats
                c_data.append(row)
        res = find_3d_centers_in_frame_simple_z(c_image, c_data, dist_threshold)
        channel2centers[c] = res
    return channel2centers


def filter_signals_in_image_by_background_contrast(
    tzc_image,
    data,
    factor=5,
    # low_bg_pixel_num_threshold=10,
    signal_size=None,
):
    """
    input: image, dicts of centers, dicts of stats, low/high threshold;
    return: a list of (center, stat)
    """
    res_centers = []
    res_stats = []
    centers, stats = data
    for i in range(len(centers)):
        center = centers[i]
        center = [int(x) for x in center]
        stat = list(stats[i])
        # center_intensity = stat[0]
        # bg_avg, bg_pixel_num = stat[2], stat[3]
        # center_intensity = tzc_image[int(center[0]), int(center[1])]
        center_intensity, center_std = utils.get_2d_center_avg(tzc_image, center, [signal_size, signal_size])
        bg_avg, bg_std = utils.get_2d_center_bg_avg(tzc_image, center, [25, 25], signal_size=signal_size)
        # bg pixel num should be fine: we have filtered by group size before.
        ratio = center_intensity / bg_avg
        if ratio >= factor:
            res_centers.append(center)
            # stat[utils.CENTER_INTENSITY_IND] = center_intensity
            # stat[utils.BACKGROUND_AVERAGE] = bg_avg
            stat.append(ratio)
            stat.append(center_intensity)
            stat.append(bg_avg)
            stat.append(bg_std)
            res_stats.append(stat)

    return res_centers, res_stats


def filter_signals_in_image_by_groupsize(image, data, low=50, high=10 ** 4):
    """
    input: image, dicts of centers, dicts of stats, low/high threshold;
    return: a list of (center, stat)
    """
    res_centers = []
    res_stats = []
    centers, stats = data
    for i in range(len(centers)):
        center = centers[i]
        stat = stats[i]
        group_size = stat[1]
        if group_size >= low and group_size <= high:
            res_centers.append(center)
            res_stats.append(stat)
    return res_centers, res_stats


def filter_signals_in_image_by_local_threshold(image, data, block_size=51, offset=20, factor=1.1):
    """
    input: image, dicts of centers, dicts of stats, low/high threshold;
    return: a list of (center, stat)
    """

    res_centers = []
    res_stats = []
    centers, stats = data
    threshold = threshold_local(image, block_size=block_size, offset=offset)

    for i in range(len(centers)):
        center = centers[i]
        stat = stats[i]
        # crop_id = stat[utils.CROP_STAT_IND]
        center_intensity = stat[utils.CENTER_INTENSITY_IND]
        if center_intensity >= factor * image[int(center[0]), int(center[1])]:
            res_centers.append(center)
            res_stats.append(stat)

    return res_centers, res_stats


def filter_signals_by_neighbors(data, cell_radius, max_allowed_signals_per_cell=utils.max_allowed_signals_per_cell):
    res_centers = []
    res_stats = []
    centers, stats = data
    zscores = [stat[utils.BC_ZSCORE_IND] for stat in stats]

    for i, center in enumerate(centers):
        stat = stats[i]
        temp_centers = []
        temp_zscores = []
        my_id = None
        for j, other in enumerate(centers):
            if i == j:
                cur_id = len(temp_centers)
                temp_centers.append(other)
                temp_zscores.append(zscores[j])

            if distance(center, other) < cell_radius:
                temp_centers.append(other)
                temp_zscores.append(zscores[j])

        if len(temp_centers) >= max_allowed_signals_per_cell:
            indices = [x for x in range(len(temp_centers))]
            sorted_indices = sorted(indices, key=lambda x: temp_zscores[x], reverse=True)
            rank = sorted_indices.index(cur_id)  # find current center's rank
            if rank > max_allowed_signals_per_cell:
                continue

        res_centers.append(center)
        res_stats.append(stat)
    # print('filter based on crop center rank, before:%d, after:%d'%(len(centers), len(res_centers)))
    return res_centers, res_stats


def filter_all_image_centers(
    images,
    # bg_corrected_images,
    tzc_centers,
    tzc_stats,
    contrast_factor=10,
    signal_size=5,
):
    """
    return: tuple of (tzc_centers, tzc_stats)
    """
    res_tzc_centers = {}
    res_tzc_stats = {}
    centers, stats = [], []
    for t, z, c in tzc_centers:
        centers = tzc_centers[t, z, c]
        stats = tzc_stats[t, z, c]
        tzc_image = images[t, z, c, :, :]
        # tzc_image = bg_corrected_images[t, z, c, :, :]
        tzc_image = bg_correction_whole_image(tzc_image, centers, stats)
        # plt.imshow(tzc_image)
        # plt.show()
        filtered_data = (centers, stats)
        # filtered_data = filter_signals_in_image_by_groupsize(tzc_image, (centers, stats))
        filtered_data = filter_signals_in_image_by_background_contrast(
            tzc_image, filtered_data, factor=contrast_factor, signal_size=signal_size
        )
        # filtered_data = filter_signals_in_image_by_local_threshold(tzc_image, (centers, stats))
        res_tzc_centers[t, z, c] = filtered_data[0]
        res_tzc_stats[t, z, c] = filtered_data[1]

    return res_tzc_centers, res_tzc_stats


def get_center_mask(images):
    n_segments = 10
    mask = skimage.segmentation.slic(images, n_segments=n_segments)
    return mask


def get_area(bd):
    """
    return nd area
    """
    dims = int(len(bd) / 2)
    res = 1
    for dim in range(dims):
        res = res * (bd[dim * 2 + 1] - bd[dim * 2])
    return res


def find_box(pixels):
    res = []
    dims = pixels.shape[1]
    boundary = []
    for dim in range(dims):
        dmin, dmax = np.min(pixels[:, dim]), np.max(pixels[:, dim])
        boundary.extend([dmin, dmax])
    return boundary


def find_boxes_from_mask(mask):
    labels = set(mask.flatten())
    boundaries = []
    # for label in labels:
    #     pixels = np.argwhere(mask == label)
    #     boundary = find_box(pixels)
    #     boundaries.append(boundary)
    label2index = {}
    for label in labels:
        label2index[label] = []
    for index, x in np.ndenumerate(mask):
        label2index[x].append(index)

    for label in labels:
        pixels = np.array(label2index[label])
        boundary = find_box(pixels)
        boundaries.append(boundary)

    return boundaries


def filter_cell_boxes(boundaries):
    low_threshold = 1000
    high_threshold = 500 ** 2
    res = []
    for boundary in boundaries:
        area = get_area(boundary)
        if area < low_threshold or area > high_threshold:
            continue
        res.append(np.array(boundary))
    return res


def combine_centers_within_same_t(images, centers_data, dist_threshold, max_center_num=None):
    """
    combine signal centers found in 2d images and convert these centers to 3d ver.
    centers_data: a tuple of (centers, stats), two lists of centers, stats
    return: t x c x (avgCenters, maxCenters, stats)
    """
    # centers, stats = read_centers_data(center_path)
    centers, stats = centers_data
    all_t, all_z, all_c = set(), set(), set([c for c in range(images.shape[2])])
    for t, z, c in centers:
        all_t.add(t)
        all_z.add(z)

    t2data = {}
    t_max = max(all_t)
    assert len(all_t) == t_max + 1
    t_centers_data = [None for _ in range(len(all_t))]
    for t in all_t:
        t2data[t] = []
        for z in all_z:
            for c in all_c:
                if (t, z, c) in centers:
                    t2data[t].append((z, c, centers[t, z, c], stats[t, z, c]))
        image = images[t, :, :, :, :]
        channel2data = find_3d_centers_in_frame_simple(image, t2data[t], dist_threshold)
        if max_center_num:
            for c in channel2data:
                # last entry is center2stats
                for i in range(0, len(channel2data[c]) - 1):
                    channel2data[c][i] = channel2data[c][i][:max_center_num]
        t_centers_data[t] = channel2data

    return t_centers_data


def combine_prob_map_centers(tzc2centers, tzc2probMaps, threshold=utils.collapse_dist_threshold):
    t_max, z_max, c_max = -1, -1, -1
    for t, z, c in tzc2centers:
        t_max = max(t_max, t)
        z_max = max(z_max, z)
        c_max = max(c_max, c)
    res = {}  # results
    for c in range(c_max + 1):
        for t in range(t_max + 1):
            z_centers = []
            z_maps = []
            for z in range(z_max + 1):
                # in case some z data are missing
                if (
                    t,
                    z,
                    c,
                ) not in tzc2centers:
                    continue
                centers = tzc2centers[t, z, c]
                centers = [list(x) + [z] for i, x in enumerate(centers)]  # convert to 3d
                probMap = tzc2probMaps[t, z, c]
                z_maps.append(probMap)
                z_centers.extend(centers)

            # collapse z centers
            z_centers = sorted(z_centers, key=lambda x: z_maps[x[2]][x[0], x[1], 0], reverse=True)  # 0 for signal prob
            groups = [[center] for center in z_centers]
            res_centers, res_groups = collapse_centers(
                z_centers, groups, dist_threshold=utils.collapse_dist_threshold_3d
            )
            res[t, c] = res_centers
    return res


def get_cropped_images(images, bd):
    """
    Use numpy broadcast to get crop images along last dim axis.
    :param images: 3d or 2d images
    :param bd: a list of boundary min and max [min0, max0, min1, max1, ....]
    :return: a list of cropped images [image0, image1, image2, ...]
    """
    ind_tuple = []
    dims = int(np.array(bd).shape[0] / 2)
    for dim in range(dims):
        ind_tuple.append(slice(bd[dim * 2], bd[dim * 2 + 1] + 1, 1))
    # print('crop slices:', ind_tuple)
    return images[tuple(ind_tuple)]


def get_3d_mask_felzenszwalb(images):
    """
    3d version of felzenzwalb simply by apply felzenswalb to each
    2d image and make label in each 2d image unique.
    not a true "3d fel" segmentation.
    :param images:
    :return:
    """
    mask = []
    label_num = 0
    for z in range(images.shape[2]):
        print("felzenszwalb handling %d of %d images" % (z, images.shape[2]))
        image = images[:, :, z]
        z_mask = skimage.segmentation.felzenszwalb(image, scale=20)
        z_mask += label_num
        label_num += len(set(z_mask.flatten()))
        mask.append(z_mask)
    mask = np.moveaxis(np.array(mask), 0, -1)
    return mask


def find_boxes(images):
    """
    utility to segment and label images
    :param images:
    :return: 3d boundaries and mask, mask: same shape as images
    """
    assert len(images.shape) == 3
    n_segments = 100
    knn_input_images = images.reshape(list(images.shape) + [1])
    # mask = skimage.segmentation.slic(knn_input_images,
    #                                  n_segments=n_segments,
    #                                  compactness=0.01,
    #                                  enforce_connectivity=False)
    mask = get_3d_mask_felzenszwalb(images)
    # mask = skimage.segmentation.watershed(images)
    labels = set(mask.flatten())
    print("segmentation found %d segments" % len(labels))
    # print('mask shape:', mask.shape)
    bds = find_boxes_from_mask(mask)
    bds = filter_cell_boxes(bds)
    print("segmentation found %d segments after filtering" % len(bds))
    return bds, mask


def get_all_cropped_images(images, bds):
    res = []
    for bd in bds:
        # print('bd:', bd)
        cropped_images = np.array(get_cropped_images(images, bd))
        res.append(cropped_images)
    return res


def generate_crops(images):
    """
    images: 3d images (x, y, z), 1 channel
    return: a list of cropped 3d images, bds, 3d_mask
    """

    bds, mask = find_boxes(images)
    cropped_images_list = get_all_cropped_images(images, bds)
    return cropped_images_list, bds, mask


def max_project_images_2d(images):
    """
    input: images: t x z x c x X x y
    return: t x c x X x Y
    """
    ts, zs, cs, xs, ys = images.shape
    projection = np.zeros(
        (
            ts,
            cs,
            xs,
            ys,
        )
    )
    for t in range(ts):
        for c in range(cs):
            images_3d = images[t, :, c, :, :]
            projection[t, c, :, :] = np.max(images_3d, axis=0)
    return projection


def max_project_images_xyz(images):
    """
    :param images:X x Y x Z
    :return: X x Y projection
    """
    projection = np.max(images, axis=2)
    return projection


def filter_regions_by_area(regions, min_area, max_area):
    res = []
    for region in regions:
        area = region.filled_area
        if area < min_area or area > max_area:
            continue
        res.append(region)
    return res


def find_cells_directly_in_prob_map(prob_map):
    if len(prob_map.shape) == 3:
        prob_map = prob_map[..., 2]
    if len(prob_map.shape) != 2:
        assert False

    prob_map[prob_map < 0.5] = 0
    cell_seg = prob_map
    cell_seg_labels = skimage.segmentation.felzenszwalb(cell_seg, scale=20000)
    # cell_seg_labels = skimage.segmentation.slic(cell_seg.astype(np.double),
    #                                             n_segments=700)
    regions = measure.regionprops(cell_seg_labels, intensity_image=cell_seg)
    regions = filter_regions_by_area(regions, min_cell_area, max_cell_area)
    return cell_seg_labels, regions


def find_signal_directly_in_prob_map(prob_map):
    if len(prob_map.shape) == 3:
        prob_map = prob_map[..., 0]
    if len(prob_map.shape) != 2:
        assert False
    prob_map[prob_map < 0.5] = 0
    coords = skimage.feature.peak_local_max(prob_map, min_distance=utils.collapse_dist_threshold, indices=True)
    coords = sorted(coords, key=lambda x: prob_map[x[0], x[1]], reverse=True)
    groups = [[center] for center in coords]
    coords, groups = collapse_centers(coords, groups, dist_threshold=utils.collapse_dist_threshold)
    return coords


def find_signal_directly_in_tzc_prob_map(tzc_map):
    res = {}
    for t, z, c in tzc_map:
        res[t, z, c] = find_signal_directly_in_prob_map(tzc_map[t, z, c])
    return res


def find_cells_directly_in_tzc_prob_map(tzc_map):
    res_regions = {}
    res_label_masks = {}
    for t, z, c in tzc_map:
        res_label_masks[t, z, c], res_regions[t, z, c] = find_cells_directly_in_prob_map(tzc_map[t, z, c][..., 2])

    return res_label_masks, res_regions


def segment_image_to_small_pieces(image, sh, sw, return_offset=False, start_h_offset=0, start_w_offset=0):
    """

    :param image:
    :param h: smaller h expected
    :param w:
    :return: offset: offset in original image space
    """
    h, w = image.shape[:2]
    res = []
    offsets = []
    # for i in range(start_h_offset, h - sh + 1, sh):
    #     for j in range(start_w_offset, w - sw + 1, sw):
    for i in range(start_h_offset, h, sh):
        for j in range(start_w_offset, w, sw):

            if i + sh <= h and j + sw <= w:
                res.append(image[i : i + sh, j : j + sw, ...])
                offsets.append([i, j])
            else:
                h_lb, h_ub, w_lb, w_ub = i, i + sh, j, j + sw
                if i + sh > h:
                    h_lb = h - sh
                    h_ub = h
                if j + sw > w:
                    w_lb = w - sw
                    w_ub = w
                res.append(image[h_lb:h_ub, w_lb:w_ub, ...])
                offsets.append([h_lb, w_lb])
    if return_offset:
        return res, offsets
    return res


def segment_images_to_small_pieces(images, sh, sw, return_offset=False):
    res = []
    offsets = []
    images = np.array(images)
    for i in range(len(images)):
        # print(images.shape)
        if not return_offset:
            smaller_images = segment_image_to_small_pieces(images[i, ...], sh, sw, return_offset=return_offset)
            res.extend(smaller_images)
        else:
            smaller_images, smaller_offsets = segment_image_to_small_pieces(
                images[i, ...], sh, sw, return_offset=return_offset
            )
            res.extend(smaller_images)
            offsets.append(smaller_offsets)

    if return_offset:
        return np.array(res), np.array(offsets)
    return np.array(res)


def map_single_frame_signal_coord_to_cells(signal_coord, cur_t, cell_mappings):
    """
    returns: cell id, time at which this cell (nearest)
    """
    for t in range(cur_t, -1, -1):
        cell_map = cell_mappings[t]
        for cell_id in cell_map:
            cell = cell_map[cell_id]
            if cell.is_in_cell(signal_coord):
                return cell
    return None


def map_signal_ids_to_cells(signal_mappings, cell_mappings):
    assert len(signal_mappings) == len(cell_mappings)
    T = len(signal_mappings)
    t_trackId2cell = []
    for t in range(T):
        signal_map = signal_mappings[t]
        t_trackId2cell.append({})
        for signal_id in signal_map:
            coord = signal_map[signal_id]
            cell = map_single_frame_signal_coord_to_cells(coord, t, cell_mappings)
            t_trackId2cell[t][signal_id] = cell
    return t_trackId2cell


def normalize_images(Y):
    """
    :param Y: images, N x w x h x n_class
    :return: a normalized version of Y
    """
    Y = np.array(Y)
    res = []
    channel_num = Y.shape[-1]
    for i in range(len(Y)):
        y = Y[i]
        # normalize channels separately
        y_re = y.reshape((-1, y.shape[-1]), order="F")
        mean = np.mean(y_re, axis=0)
        std = np.std(y_re, axis=0)
        y = (y - mean) / std
        res.append(y)
    return np.array(res)


def resize_image_to_2_powers(image):
    closest_size = 2 ** int(math.ceil(math.log2(image.shape[0])))
    # closest_size = 2 ** int(math.floor(math.log2(image.shape[0])))
    resized_img = skimage.transform.resize(image, (closest_size, closest_size))
    return resized_img


def get_CNN_regression_mask(image, relu_edt=False):
    normalized_image = normalize_images([image])[0]
    resized_img = resize_image_to_2_powers(normalized_image)
    predicted_mask = config.model.predict(np.array([resized_img[..., np.newaxis]]), batch_size=1)[0]
    print("predicted map shape:", predicted_mask.shape)
    predicted_mask = predicted_mask.reshape(predicted_mask.shape[:-1])
    predicted_mask = skimage.transform.resize(predicted_mask, image.shape)
    # try old felzenswalb
    # mask = skimage.segmentation.felzenszwalb(predicted_mask, scale=20, min_size=utils.min_cell_area)
    # watershed segmentation
    edt_dist = None
    if not relu_edt:
        possible_cell_mask = predicted_mask > 0.5  # threshold
        edt_dist = ndi.distance_transform_edt(possible_cell_mask)
    else:
        possible_cell_mask = predicted_mask
        edt_dist = predicted_mask
    markers = skimage.feature.peak_local_max(edt_dist, min_distance=20, threshold_abs=3, indices=False)
    markers = ndi.label(markers)[0]
    watershed_mask = skimage.segmentation.watershed(-edt_dist, markers=markers, mask=possible_cell_mask)

    # simple visualization check
    # utils.show_images([image, normalized_image, resized_img, predicted_mask, mask, possible_cell_mask, watershed_mask])
    return watershed_mask


def find_cells_directly_in_max_projected_nd2(nd2_path, save_dir="./test_figs", return_masks=False):
    """
    :param nd2_path: target nd2 file to process.
    :param save_dir: directory for saving figures
    :return: a dictionary: from (t, c) to a list of regions in specific time and channel
    """
    utils.make_dir(save_dir, abort=False)
    images = utils.ND2Reader(nd2_path)
    z_levels = images.metadata["z_levels"]
    frames = images.metadata["frames"]
    t_num = len(frames)
    channel_num = len(images.metadata["channels"])
    tc_projections = {}
    tc_masks = {}
    tc_regions = {}
    utils.make_dir(save_dir, abort=False)
    for c in range(channel_num):
        for t in frames:
            mask_path = os.path.join(save_dir, "C%d_T%d_mask" % (c, t))
            print("channel:", c, t, "/", len(frames))
            images = utils.read_nd2_along_z(nd2_path, c, t)
            # print('test images shape:', images.shape)
            projection = max_project_images_xyz(images)
            # projection = skimage.restoration.unsupervised_wiener(projection, psf)
            tc_projections[t, c] = projection

            mask = None
            if os.path.exists(mask_path):
                mask = utils.read_mask(mask_path)
                print("using existing mask")
            else:
                mask = get_CNN_regression_mask(projection)
                # mask = skimage.segmentation.felzenszwalb(projection, scale=3, min_size=utils.min_cell_area)
                # mask = skimage.segmentation.quickshift(projection)
            tc_masks[t, c] = mask
            regions = skimage.measure.regionprops(mask, intensity_image=projection)
            regions = filter_regions_by_area(regions, min_cell_area, max_cell_area)
            tc_regions[t, c] = regions

            # save masks if needed
            fig, axes = plt.subplots(2)
            axes[0].imshow(projection)
            axes[1].imshow(mask)
            plt.suptitle("c=%d,t=%d" % (c, t))
            # plt.show()
            plt.savefig(os.path.join(save_dir, "C%d_T%d_plot" % (c, t)))
            plt.close()
            utils.save_mask(mask, mask_path)

    if return_masks:
        return tc_regions, tc_masks
    return tc_regions


def correct_tiling_gap(image, row_num=3, col_num=3):
    height, width = image.shape[:2]
    tile_h, tile_w = height // row_num, width // col_num
    bg_corrected_img = bg_correction(image)
    return bg_corrected_img


if __name__ == "__main__":
    # BGR order
    # boundary = [(0, 0, 100), (50, 50, 255)]
    # mask_edge_example(img_path, boundary`)
    pass
