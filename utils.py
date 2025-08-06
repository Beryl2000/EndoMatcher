import cv2
import numpy as np
import torch
import tqdm
import os
from natsort import ns, natsorted
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib

import dataset


def read_color_img(image_path,out_size):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img



def type_float_and_reshape(array, shape):
    array = array.astype(np.float32)
    return array.reshape(shape)


def read_visible_view_indexes(prefix_seq):
    path = prefix_seq / 'visible_view_indexes'
    if not path.exists():
        return []

    visible_view_indexes = []
    with open(str(path)) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))
    return visible_view_indexes


def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))
    return selected_indexes



def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path)
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        lists_3D_points.append([temp[0], temp[1], temp[2], 1.0])
    return lists_3D_points


def read_view_indexes_per_point(prefix_seq, visible_view_indexes,
                                point_cloud_count):
    view_indexes_per_point = np.zeros(
        (point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(str(prefix_seq / 'view_indexes_per_point')) as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(
                    int(line))] = 1
    return view_indexes_per_point



def overlapping_visible_view_indexes_per_point(visible_view_indexes_per_point, visible_interval):
    temp_array = np.copy(visible_view_indexes_per_point) 
    view_count = visible_view_indexes_per_point.shape[1]
    for i in range(view_count):
        visible_view_indexes_per_point[:, i] = \
            np.sum(temp_array[:, max(0, i - visible_interval):min(view_count, i + visible_interval)], axis=1)

    return visible_view_indexes_per_point 


def read_pose_data(prefix_seq):
    stream = open(str(prefix_seq / "motion.yaml"), 'r')
    doc =yaml.load(stream,Loader=yaml.FullLoader)
    keys, values = doc.items()
    poses = values[1]
    return poses



def quat_to_pos_matrix_hm(p_x, p_y, p_z, x, y, z, w):
    T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)

    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z - w * x)

    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T


def quat_to_pos_matrix_JPL(p_x, p_y, p_z, x, y, z, w):
    T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y + w * z)
    T[0, 2] = 2 * (x * z - w * y)

    T[1, 0] = 2 * (x * y - w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z + w * x)

    T[2, 0] = 2 * (x * z + w * y)
    T[2, 1] = 2 * (y * z - w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count,system=0):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        p_x ,p_y,p_z = poses["poses[" + str(i) + "]"]['position']['x'],poses["poses[" + str(i) + "]"]['position']['y'],poses["poses[" + str(i) + "]"]['position']['z']

        w,x ,y ,z = poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],poses["poses[" + str(i) + "]"]['orientation']['y'],poses["poses[" + str(i) + "]"]['orientation']['z']
        
        if system==0:
            transform = quat_to_pos_matrix_hm(p_x, p_y, p_z, x, y, z, w) 
        else: transform = quat_to_pos_matrix_JPL(p_x, p_y, p_z, x, y, z, w)

        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))

    return extrinsic_matrices, projection_matrices


def global_scale_estimation(extrinsics, point_cloud):
    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)

    for i, extrinsic in enumerate(extrinsics):
        if i == 0:
            max_bound = extrinsic[:3, 3]
            min_bound = extrinsic[:3, 3]
        else:
            temp = extrinsic[:3, 3]
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_1 = np.linalg.norm(max_bound - min_bound, ord=2)

    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)
    for i, point in enumerate(point_cloud):
        if i == 0:
            max_bound = np.asarray(point[:3], dtype=np.float32)
            min_bound = np.asarray(point[:3], dtype=np.float32)
        else:
            temp = np.asarray(point[:3], dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_2 = np.linalg.norm(max_bound - min_bound, ord=2)

    return max(norm_1, norm_2)



def get_color_imgs(prefix_seq, visible_view_indexes,image_size,out_size):
    imgs = []
    for i in visible_view_indexes:
        file_path = os.path.join(prefix_seq, "{:05d}".format(i))
        if os.path.exists(file_path + ".png"):
            img = cv2.imread(file_path + ".png")
        elif os.path.exists(file_path + ".jpg"):
            img = cv2.imread(file_path + ".jpg")
        else:
            print(f"No image found for index {i}")
            img = None

        downsampled_img = cv2.resize(img, (0, 0), fx=out_size[1]/image_size[1], fy=out_size[0]/image_size[0])
        imgs.append(downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def compute_sanity_threshold(sanity_array, inlier_percentage):
    hist, bin_edges = np.histogram(sanity_array, bins=np.arange(1000) * np.max(sanity_array) / 1000.0,
                                   density=True)
    histogram_percentage = hist * np.diff(bin_edges)
    percentage = inlier_percentage

    max_index = np.argmax(histogram_percentage)
    histogram_sum = histogram_percentage[max_index]
    pos_counter = 1
    neg_counter = 1
    while True:
        if max_index + pos_counter < len(histogram_percentage):
            histogram_sum = histogram_sum + histogram_percentage[max_index + pos_counter]
            pos_counter = pos_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index - neg_counter >= 0:
            histogram_sum = histogram_sum + histogram_percentage[max_index - neg_counter]
            neg_counter = neg_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index + pos_counter >= len(histogram_percentage) and max_index - neg_counter < 0:
            sanity_threshold_max = np.max(bin_edges)
            sanity_threshold_min = np.min(bin_edges)
            break
    return sanity_threshold_min, sanity_threshold_max


def get_clean_point_list(imgs, out_size,point_cloud, view_indexes_per_point, inlier_percentage,
                         projection_matrices,
                         extrinsic_matrices):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    assert (inlier_percentage > 0.0 and inlier_percentage <= 1.0)
    point_cloud_contamination_accumulator = np.zeros(array_3D_points.shape[0], dtype=np.int32) 
    point_cloud_appearance_count = np.zeros(array_3D_points.shape[0], dtype=np.int32) 
    height, width, channel = imgs[0].shape
    valid_frame_count = 0
    mask_boundary=np.ones(out_size)*255
    # mask_boundary[10:246,10:246]=0
    mask_boundary = mask_boundary.reshape((-1, 1))
    for i in range(len(projection_matrices)): 
        img = imgs[i]
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]
        img = np.array(img, dtype=np.float32) / 255.0
        img_filtered = cv2.bilateralFilter(src=img, d=7, sigmaColor=25, sigmaSpace=25) 
        img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL) 

        view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
        visible_point_indexes = np.where(view_indexes_frame > 0.5)
        visible_point_indexes = visible_point_indexes[0] 

        points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
        points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))
        points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
        points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1)) 

        visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
        visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
        indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                           (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
                           & (visible_points_3D_camera[:, 2] > 0)) 
        indexes = indexes[0] 
        in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                       np.round(visible_points_2D_image[indexes, 1]) * width).astype(np.int32).reshape((-1)) 
        temp_mask = mask_boundary[in_image_point_1D_locations, :]
        indexes_2 = np.where(temp_mask[:, 0] == 255)
        indexes_2 = indexes_2[0]
        in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2] 

        points_depth = visible_points_3D_camera[indexes[indexes_2], 2] 
        img_hsv = img_hsv.reshape((-1, 3))
        points_brightness = img_hsv[in_mask_point_1D_locations, 2]
        sanity_array = points_depth ** 2 * points_brightness
        point_cloud_appearance_count[visible_point_indexes[indexes[indexes_2]]] += 1 
        if sanity_array.shape[0] < 2:
            continue
        valid_frame_count += 1
        sanity_threshold_min, sanity_threshold_max = compute_sanity_threshold(sanity_array, inlier_percentage)
        indexes_3 = np.where((sanity_array <= sanity_threshold_min) | (sanity_array >= sanity_threshold_max)) 
        indexes_3 = indexes_3[0] 
        point_cloud_contamination_accumulator[visible_point_indexes[indexes[indexes_2[indexes_3]]]] += 1

    clean_point_cloud_array = (point_cloud_contamination_accumulator < point_cloud_appearance_count / 2).astype(
        np.float32) 
    print("{} points eliminated".format(int(clean_point_cloud_array.shape[0] - np.sum(clean_point_cloud_array)))) 
    return clean_point_cloud_array


def feature_matching_single_generation(feature_map_1, feature_map_2,
                                       kps_1D_1, cross_check_distance, device):
    with torch.no_grad():
        feature_length, height, width = feature_map_1.shape

        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(device=device).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)
        del sampled_feature_vectors_2, source_filter_response_map, feature_1d_locations_2

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(device=device).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()


        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())

        valid_detected_target_2d_locations = torch.cat(
            [torch.fmod(valid_detected_1d_locations_2.float(), width).view(-1, 1),
             torch.floor(valid_detected_1d_locations_2.float() / width).view(-1, 1)],
            dim=1).view(-1, 2).float()
        valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
        valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()
        return valid_source_keypoint_indexes, valid_detected_target_2d_locations


def extract_keypoints(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    keypoints_list_2D = []

    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = cv2.cvtColor(
            np.uint8(255 * ((color_1 - color_1.min()) /
                            (color_1.max() - color_1.min()))),
            cv2.COLOR_RGB2BGR)
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)

        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = np.zeros((len(kps)))
        temp_2d = np.zeros((len(kps), 2))

        for j, point in enumerate(kps):
            temp[j] = np.round(point.pt[0]) + np.round(point.pt[1]) * width
            temp_2d[j, 0] = np.round(point.pt[0])
            temp_2d[j, 1] = np.round(point.pt[1])

        keypoints_list_1D.append(temp)
        keypoints_list_2D.append(temp_2d)
    return keypoints_list, keypoints_list_1D, keypoints_list_2D, descriptions_list


def gather_feature_matching_data_new_cross(sub_folder, data_root,input_size,
                                 network_downsampling, load_intermediate_data, precompute_root,
                                 batch_size,device):

    video_frame_filenames = []
    list = os.listdir(str(sub_folder / 'images'))
    files = natsorted(list, alg=ns.PATH)
    for filename in files:
        video_frame_filenames.append(Path(os.path.join(str(sub_folder / 'images'), filename)))
    print("Gathering feature matching data for {}".format(str(sub_folder)))
    folder_list =[data_root]
    video_dataset = dataset.MixDataset(image_file_names=video_frame_filenames,
                                       folder_list=folder_list,
                                       out_size=input_size,
                                       network_downsampling=network_downsampling,
                                       load_intermediate_data=load_intermediate_data,
                                       intermediate_data_root=precompute_root,
                                       phase="image_loading")
    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    with torch.no_grad():
        tq = tqdm.tqdm(total=len(video_loader) * batch_size)
        for batch, (colors_1, boundaries, image_names,
                    folders,pair_intrinsic_matrices) in enumerate(video_loader):
            tq.update(batch_size)
            colors_1 = colors_1.cuda(device=device)
            if batch == 0:
                boundary = boundaries[0].data.numpy()

            for idx in range(colors_1.shape[0]):
                colors_list.append(colors_1[idx].data.cpu().numpy())
    tq.close()

    return colors_list, boundary,pair_intrinsic_matrices




def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=150, path=None):

    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1.5)
                                        for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=6)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=6)

    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=50, va='top', ha='left', color=txt_color)

    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig