import cv2
import numpy as np
import torch
import tqdm
import os
from natsort import ns, natsorted
from pathlib import Path
from plyfile import PlyData
import yaml
import matplotlib.pyplot as plt
import matplotlib
import random
import kornia
import torch.nn.functional as F
import pickle
from tensorboardX import SummaryWriter


import dataset


def set_random_seed(seed=10085):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prepare_paths(cfg):
    if not cfg.training_data_root.exists():
        raise IOError("Specified training data root does not exist.")
    return cfg.log_root, cfg.training_data_root, cfg.precompute_root


def init_tensorboard(log_root):
    writer = SummaryWriter(logdir=str(log_root))
    print(f"Created TensorBoard visualization at {log_root}")
    return writer


def load_dataset_split():
    with open('dataset_split.pkl', 'rb') as f:
        split_list = pickle.load(f)
    return split_list

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

        max_responses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_responses.view(-1)
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

        max_responses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
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




def get_color_file_names_by_bag(root, training_patient_id, validation_patient_id, testing_patient_id):
    training_image_list = []
    validation_image_list = []
    testing_image_list = []

    if not isinstance(training_patient_id, list):
        training_patient_id = [training_patient_id]
    if not isinstance(validation_patient_id, list):
        validation_patient_id = [validation_patient_id]
    if not isinstance(testing_patient_id, list):
        testing_patient_id = [testing_patient_id]

    for id in training_patient_id:
        training_image_list += list(root.glob('{:d}/*/images/0*.[jp][pn]g'.format(id)))
    for id in testing_patient_id:
        testing_image_list += list(root.glob('{:d}/*/images/0*.[jp][pn]g'.format(id)))
    for id in validation_patient_id:
        validation_image_list += list(root.glob('{:d}/*/images/0*.[jp][pn]g'.format(id)))

    training_image_list.sort()
    testing_image_list.sort()
    validation_image_list.sort()
    return training_image_list, validation_image_list, testing_image_list



def get_parent_folder_names(root, id_range):
    folder_list = []
    for id in id_range:
        folder_list += list(root.glob('{:d}/*/'.format(id)))
    folder_list.sort()
    return folder_list



def save_model(model, optimizer, epoch, step, model_path, validation_loss):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'validation': validation_loss
    }, str(model_path))
    return


def get_torch_training_data_feature_matching(height, width,camera_intrinsic, pair_projections, pair_extrinsics,pair_indexes, out_size,point_cloud,
                                             view_indexes_per_point, clean_point_list,
                                             visible_view_indexes,reprojection_error_threshold):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]

    visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]

    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0))
    in_image_indexes_1 = in_image_indexes_1[0]

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0))
    in_image_indexes_2 = in_image_indexes_2[0]

    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))

    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))

    mask_boundary=np.ones(out_size)*255
    mask_boundary = mask_boundary.reshape((-1, 1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]

    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]

    common_visible_point_indexes = list(
        np.intersect1d(np.asarray(visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]]),
                       np.asarray(visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]]), assume_unique=True))

    feature_matches = np.concatenate(
        [points_2D_image_1[common_visible_point_indexes, :2], points_2D_image_2[common_visible_point_indexes, :2]],
        axis=1)

    T_i = pair_extrinsics[0]
    T_j = pair_extrinsics[1]
    T_i_inv = np.linalg.inv(T_i)
    T_j_to_i = T_i_inv @ T_j
    R_j_to_i = T_j_to_i[:3, :3]
    t_j_to_i = T_j_to_i[:3, 3]

    K_modified=camera_intrinsic
    p1 = feature_matches[:,0:2]
    p2 = feature_matches[:,2:4]
    # Convert to normalized coordinates
    pts1_norm = np.linalg.inv(K_modified) @ np.vstack((p1.T, np.ones(p1.shape[0])))
    pts2_norm = np.linalg.inv(K_modified) @ np.vstack((p2.T, np.ones(p2.shape[0])))
    # Triangulate points (adjust as needed)
    X = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R_j_to_i,t_j_to_i)), pts1_norm[:2], pts2_norm[:2])
    X /= X[3]  # Convert from homogeneous to 3D coordinates
    points_2d = K_modified @ X[:3]
    points_2d /= points_2d[2]  # Normalize
    valid_indices = []
    for i in range(len(p1)):
        reprojection_error = np.linalg.norm(p1[i] - points_2d[:2, i])
        if reprojection_error < reprojection_error_threshold:
            valid_indices.append(i)
    pts1 = p1[valid_indices]
    pts2 = p2[valid_indices]
    feature_matches=np.concatenate([pts1, pts2],  axis=1)

    return feature_matches




def generate_heatmap_from_locations(feature_2D_locations, height, width, sigma):
    sample_size, _ = feature_2D_locations.shape

    feature_2D_locations = np.reshape(feature_2D_locations, (sample_size, 4))

    source_heatmaps = []
    target_heatmaps = []

    sigma_2 = sigma ** 2
    for i in range(sample_size):
        x = feature_2D_locations[i, 0]
        y = feature_2D_locations[i, 1]

        x_2 = feature_2D_locations[i, 2]
        y_2 = feature_2D_locations[i, 3]

        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), sparse=False, indexing='ij')

        source_grid_x = x_grid - x
        source_grid_y = y_grid - y

        target_grid_x = x_grid - x_2
        target_grid_y = y_grid - y_2

        heatmap = np.exp(-(source_grid_x ** 2 + source_grid_y ** 2) / (2.0 * sigma_2))
        heatmap_2 = np.exp(-(target_grid_x ** 2 + target_grid_y ** 2) / (2.0 * sigma_2))

        source_heatmaps.append(heatmap)
        target_heatmaps.append(heatmap_2)

    source_heatmaps = np.asarray(source_heatmaps, dtype=np.float32).reshape((sample_size, height, width))
    target_heatmaps = np.asarray(target_heatmaps, dtype=np.float32).reshape((sample_size, height, width))

    return source_heatmaps, target_heatmaps




def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    visible_view_idx = idx % len(visible_view_indexes)

    adjacent_range_list = []
    adjacent_range_list.append(adjacent_range[0])
    adjacent_range_list.append(adjacent_range[1])

    if len(visible_view_indexes) <= 2 * adjacent_range_list[0]:
        adjacent_range_list[0] = len(visible_view_indexes) // 2

    if visible_view_idx <= adjacent_range_list[0] - 1:
        increment = random.randint(adjacent_range_list[0],
                                   min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range_list[0]:
        increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range_list[0],
                                       min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    return [visible_view_idx, increment]



def read_camera_intrinsic_per_view(prefix_seq):
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(str(prefix_seq / 'camera_intrinsics_per_view')) as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                param_count += 1
            elif param_count == 1:
                temp_camera_intrincis[1][1] = float(line)
                param_count += 1
            elif param_count == 2:
                temp_camera_intrincis[0][2] = float(line)
                param_count += 1
            elif param_count == 3:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0
    return camera_intrinsics


def modify_camera_intrinsic_matrix(intrinsic_matrix,image_size,out_size):
    intrinsic_matrix_modified = np.copy(intrinsic_matrix)
    intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / (image_size[1]/out_size[1])
    intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / (image_size[0]/out_size[0])
    intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / (image_size[1]/out_size[1])
    intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / (image_size[0]/out_size[0])
    return intrinsic_matrix_modified


def get_pair_color_imgs(prefix_seq, pair_indexes,out_size):
    imgs = []
    for i in pair_indexes:
        file_path = os.path.join(prefix_seq, "{:05d}".format(i))
        if os.path.exists(file_path + ".png"):
            img = cv2.imread(file_path + ".png")
        elif os.path.exists(file_path + ".jpg"):
            img = cv2.imread(file_path + ".jpg")
        else:
            print(f"No image found for index {i}")
            img = None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        imgs.append(img)

    height, width, channel = imgs[0].shape
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs = imgs.reshape((-1, height, width, channel))
    return imgs



def sample_homography(shape, device):
    h, w = shape
    corners = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
    rg = max(h, w)
    warp = np.random.randint(-rg//10, rg//10, size=(4, 2)).astype(np.float32)
    if random.random() < 0.2:
        warp = np.random.randint(-5, 5, size=(4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(corners, corners + warp)
    homography = torch.from_numpy(M).to(device).float()
    homography = homography.unsqueeze(0)
    return homography


def get_translation_mat(image_height, image_width, trans, transformed_corners):
    left_top_min = np.min(transformed_corners, axis=0)
    right_bottom_min = np.min(np.array([image_width, image_height]) -
                              transformed_corners,
                              axis=0)
    trans_x_value = int(np.random.uniform(0, trans) * image_width)
    trans_y_value = int(np.random.uniform(0, trans) * image_height)
    if np.random.uniform() > 0.5:  #translate x with respect to left axis
        trans_x = trans_x_value if left_top_min[0] < 0 else -trans_x_value
    else:  #translate x with respect to right axis
        trans_x = trans_x_value if right_bottom_min[0] > 0 else -trans_x_value
    if np.random.uniform() > 0.5:  #translate y with respect to top axis
        trans_y = trans_y_value if left_top_min[1] < 0 else -trans_y_value
    else:  #translate y with respect to bottom axis
        trans_y = trans_y_value if right_bottom_min[1] > 0 else -trans_y_value
    translate_mat = np.eye(3)
    translate_mat[0, 2] = trans_x
    translate_mat[1, 2] = trans_y
    return translate_mat

def get_perspective_mat(patch_ratio, image_height, image_width, trans):
    patch_ratio = 1 - random.random() * (1 - patch_ratio)

    patch_bound_w, patch_bound_h = int(patch_ratio * image_width), int(patch_ratio * image_height)
    patch_corners = np.array([[0,0], [0, patch_bound_h], [patch_bound_w, patch_bound_h], [patch_bound_w, 0]]).astype(np.float32)

    homography_matrix = sample_homography((image_height, image_width), 'cpu')[0].numpy()

    trans_patch_corners = cv2.perspectiveTransform(np.reshape(patch_corners, (-1, 1, 2)), homography_matrix).squeeze(1)
    translation_matrix = get_translation_mat(image_height, image_width, trans, trans_patch_corners)
    homography_matrix = translation_matrix @ homography_matrix
    return homography_matrix



def scale_homography(homo_matrix, src_height, src_width, dest_height, dest_width):
    """
    If src and warped image is scaled by same amount, then homography needs to changed according
    to the scale in x and y direction
    """
    scale_x = dest_width / src_width
    scale_y = dest_height / src_height
    scale_matrix = np.diag([scale_x, scale_y, 1.0])
    homo_matrix = scale_matrix @ homo_matrix @ np.linalg.inv(scale_matrix)
    return homo_matrix


def erosion2d(image, strel, origin=(0, 0), border_value=1e6):
    """
    :param image:BCHW
    :param strel: BHW
    :param origin: default (0,0)
    :param border_value: default 1e6
    :return:
    """
    image_pad = F.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = F.unfold(image_pad, kernel_size=strel.shape[1])#[B,C*sH*sW,L],L is the number of patches
    strel_flatten = torch.flatten(strel,start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    # Take maximum over the neighborhood
    result, _ = diff.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)




def compute_valid_mask(image_shape,
                       homographies,
                       erosion_radius=0,
                       device='cpu'):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
    Arguments:
        input_shape: `[H, W]`, tuple, list or ndarray
        homography: B*3*3 homography
        erosion_radius: radius of the margin to be discarded.
    Returns: mask with values 0 or 1
    """
    if len(homographies.shape) == 2:
        homographies = homographies.unsqueeze(0)
    # TODO:uncomment this line if your want to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    B = homographies.shape[0]
    img_one = torch.ones(tuple([B,3,*image_shape[0:2]]),
                         device=device,
                         dtype=torch.float32)  #B,C,H,W
    mask = kornia.geometry.transform.warp_perspective(img_one,
                                                      homographies,
                                                      tuple(image_shape[0:2]),
                                                      align_corners=True)
    mask = mask.round()  #B3HW
    if erosion_radius > 0:
        # TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_radius * 2, ) * 2)
        kernel = torch.as_tensor(kernel[np.newaxis, :, :], device=device)
        _, kH, kW = kernel.shape
        origin = ((kH - 1) // 2, (kW - 1) // 2)
        mask = erosion2d(mask, torch.flip(kernel, dims=[
            1, 2
        ]), origin=origin) + 1.  # flip kernel so perform as tf.nn.erosion2d

    return mask[:,0,:,:]
