import numpy as np
import cv2
import pickle
from multiprocessing import Process, Queue
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import torch
from pathlib import Path
import os
import random
from torchvision.transforms import Compose
import matplotlib.cm as cm

from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import utils

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)


def pre_processing_data(process_id, folder_list,out_size, inlier_percentage,
                        visible_interval,
                        queue_clean_point_list, queue_intrinsic_matrix, queue_point_cloud,
                        queue_view_indexes_per_point, queue_selected_indexes,
                        queue_visible_view_indexes,
                        queue_extrinsics, queue_projection,queue_estimated_scale):
    for folder in folder_list:
        colmap_result_folder = folder / "colmap2" / "0"
        images_folder = folder / "images"
        images_list = os.listdir(str(images_folder))
        exam = cv2.imread(str(images_folder/ images_list[0]), cv2.IMREAD_GRAYSCALE)
        image_size=exam.shape

        folder_str = str(folder)

        visible_view_indexes = utils.read_visible_view_indexes(colmap_result_folder)
        if len(visible_view_indexes) == 0:
            print("Sequence {} does not have relevant files".format(folder_str))
            continue
        queue_visible_view_indexes.put([folder_str, visible_view_indexes])

        selected_indexes = utils.read_selected_indexes(colmap_result_folder)
        queue_selected_indexes.put([folder_str, selected_indexes])

        undistorted_camera_intrinsic_per_view = utils.read_camera_intrinsic_per_view(colmap_result_folder)
        cropped_downsampled_undistorted_intrinsic_matrix = utils.modify_camera_intrinsic_matrix(
            undistorted_camera_intrinsic_per_view[0], image_size,out_size)
        queue_intrinsic_matrix.put([folder_str, cropped_downsampled_undistorted_intrinsic_matrix])

        point_cloud = utils.read_point_cloud(str(colmap_result_folder / "structure.ply"))
        queue_point_cloud.put([folder_str, point_cloud])

        view_indexes_per_point = utils.read_view_indexes_per_point(colmap_result_folder, visible_view_indexes=
        visible_view_indexes, point_cloud_count=len(point_cloud))
        view_indexes_per_point = utils.overlapping_visible_view_indexes_per_point(view_indexes_per_point,
                                                                                  visible_interval)
        queue_view_indexes_per_point.put([folder_str, view_indexes_per_point])


        poses = utils.read_pose_data(colmap_result_folder)
        visible_extrinsic_matrices, visible_cropped_downsampled_undistorted_projection_matrices = \
            utils.get_extrinsic_matrix_and_projection_matrix(poses,
                                                             intrinsic_matrix=
                                                             cropped_downsampled_undistorted_intrinsic_matrix,
                                                             visible_view_count=len(visible_view_indexes))
        queue_extrinsics.put([folder_str, visible_extrinsic_matrices])
        queue_projection.put([folder_str, visible_cropped_downsampled_undistorted_projection_matrices])

        global_scale = utils.global_scale_estimation(visible_extrinsic_matrices, point_cloud)
        queue_estimated_scale.put([folder_str, global_scale])

        visible_cropped_downsampled_imgs = utils.get_color_imgs(images_folder,visible_view_indexes,
                                                                image_size,out_size)
        clean_point_indicator_array = utils.get_clean_point_list(imgs=visible_cropped_downsampled_imgs,
                                                                 out_size=out_size,
                                                                 point_cloud=point_cloud,
                                                                 inlier_percentage=inlier_percentage,
                                                                 projection_matrices=
                                                                 visible_cropped_downsampled_undistorted_projection_matrices,
                                                                 extrinsic_matrices=visible_extrinsic_matrices,
                                                                 view_indexes_per_point=view_indexes_per_point)
        queue_clean_point_list.put([folder_str, clean_point_indicator_array])

        print("sequence {} finished".format(folder_str))
    print("{}th process finished".format(process_id))

class MixDataset(Dataset):
    def __init__(self, image_file_names, folder_list,out_size,
                 network_downsampling, load_intermediate_data,
                 intermediate_data_root, phase, train_phase,visible_interval=30, pre_workers=12, inlier_percentage=0.998,reprojection_error_threshold=5,
                 adjacent_range=(1, 1), num_iter=None,
                 sampling_size=10, heatmap_sigma=5.0,split_list=[]):

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_file_names = sorted(image_file_names)
        self.folder_list = folder_list

        self.split_list = split_list
        if self.split_list!=[]:
            self.c3vd_trval_ls=self.split_list['C3VD']['train']+self.split_list['C3VD']['val']
            self.endoslam_trval_ls=self.split_list['EndoSLAM']['train']+self.split_list['EndoSLAM']['val']
            self.scared_trval_ls=self.split_list['SCARED']['train']+self.split_list['SCARED']['val']
            self.endomapper_trval_ls=self.split_list['EndoMapper']['train']+self.split_list['EndoMapper']['val']
            self.efm_trval_ls=self.split_list['EFM_COLON']['train']+self.split_list['EFM_COLON']['val']
            self.ours_trval_ls=self.split_list['Ours']['train']+self.split_list['Ours']['val']

            self.c3vd_trval_ls = [Path(path).name for path in self.c3vd_trval_ls]
            self.endoslam_trval_ls = [Path(path).name for path in self.endoslam_trval_ls]
            self.scared_trval_ls = [Path(path).name for path in self.scared_trval_ls]
            self.endomapper_trval_ls = [Path(path).name for path in self.endomapper_trval_ls]
            self.efm_trval_ls = [Path(path).name for path in self.efm_trval_ls]
            self.ours_trval_ls = [Path(path).name for path in self.ours_trval_ls]

            self.datasets_list = [self.c3vd_trval_ls, self.endoslam_trval_ls, self.scared_trval_ls,self.endomapper_trval_ls,self.efm_trval_ls,self.ours_trval_ls]

        self.intermediate_data_root=intermediate_data_root
        assert (len(adjacent_range) == 2)
        self.adjacent_range = adjacent_range
        self.inlier_percentage = inlier_percentage 
        self.reprojection_error_threshold=reprojection_error_threshold
        self.out_size=out_size
        self.network_downsampling = network_downsampling 
        self.phase = phase
        self.train_phase = train_phase
        self.visible_interval = visible_interval 
        self.sampling_size = sampling_size
        self.num_iter = num_iter
        self.heatmap_sigma = heatmap_sigma
        self.pre_workers = min(len(folder_list), pre_workers)
        self.transform = Compose(
            [
                Resize(
                    self.out_size[0],
                    self.out_size[1],
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.clean_point_list_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}
        self.estimated_scale_per_seq = {}

        self.real_flag=-1

        self.apply_aug = True

        import albumentations as alb
        self.aug_list = [
            alb.OneOf([alb.RandomBrightness(limit=0.2, p=0.8), alb.RandomContrast(limit=0.3, p=0.8)], p=0.5),
            alb.OneOf([alb.MotionBlur(p=0.8), alb.GaussNoise(p=0.8)], p=0.8),
            ]
        self.aug_func = alb.Compose(self.aug_list, p=1)
        self.aug_list0 = [
            alb.ElasticTransform(p=1, alpha=1, sigma=60, alpha_affine=10),
            ]
        self.aug_func0 = alb.Compose(self.aug_list0, p=1)

        self.resize = True
        self.config = {
            'apply_color_aug': True,
            'image_height': self.out_size[0],
            'image_width': self.out_size[1],
            'augmentation_params':{
                'patch_ratio': 0.8,
                'translation': 0.2,
            }
        }
        self.st = 0
        self.aug_params = self.config['augmentation_params']

        if self.phase != "image_loading":

            if not any(self.intermediate_data_root.iterdir()) :
                queue_clean_point_list = Queue()
                queue_intrinsic_matrix = Queue()
                queue_point_cloud = Queue()
                queue_view_indexes_per_point = Queue()
                queue_selected_indexes = Queue()
                queue_visible_view_indexes = Queue()
                queue_extrinsics = Queue()
                queue_projection = Queue()
                queue_estimated_scale = Queue()

                process_pool = []

                interval = len(self.folder_list) / self.pre_workers

                largest_h,largest_w=self.out_size[0],self.out_size[1]
                print("Largest image size is: ", largest_h, largest_w)

                print("Start pre-processing dataset...")
                process_pool = []
                for i in range(self.pre_workers):
                    process_pool.append(Process(target=pre_processing_data,
                                                args=(i, self.folder_list[int(np.round(i * interval)):
                                                                        min(int(np.round((i + 1) * interval)),
                                                                            len(self.folder_list))],
                                                    self.out_size,
                                                    self.inlier_percentage, self.visible_interval,
                                                    queue_clean_point_list,
                                                    queue_intrinsic_matrix, queue_point_cloud,
                                                    queue_view_indexes_per_point,
                                                    queue_selected_indexes,
                                                    queue_visible_view_indexes,
                                                    queue_extrinsics, queue_projection,
                                                    queue_estimated_scale)))

                for t in process_pool:
                    t.start()

                count = 0
                folders = []
                folders_name=[]
                for t in process_pool:
                    print("Waiting for {:d}th process to complete".format(count))
                    count += 1
                    while t.is_alive():
                        while not queue_selected_indexes.empty():
                            folder, selected_indexes = queue_selected_indexes.get()
                            folders.append(folder)
                            folders_name.append('_'.join(list(Path(folder).parts[-2:])) )
                            self.selected_indexes_per_seq[folder] = selected_indexes
                        while not queue_visible_view_indexes.empty():
                            folder, visible_view_indexes = queue_visible_view_indexes.get()
                            self.visible_view_indexes_per_seq[folder] = visible_view_indexes
                        while not queue_view_indexes_per_point.empty():
                            folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                            self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
                        while not queue_clean_point_list.empty():
                            folder, clean_point_list = queue_clean_point_list.get()
                            self.clean_point_list_per_seq[folder] = clean_point_list
                        while not queue_intrinsic_matrix.empty():
                            folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                            self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
                        while not queue_extrinsics.empty():
                            folder, extrinsics = queue_extrinsics.get()
                            self.extrinsics_per_seq[folder] = extrinsics
                        while not queue_projection.empty():
                            folder, projection = queue_projection.get()
                            self.projection_per_seq[folder] = projection
                        while not queue_point_cloud.empty():
                            folder, point_cloud = queue_point_cloud.get()
                            self.point_cloud_per_seq[folder] = point_cloud
                        while not queue_estimated_scale.empty():
                            folder, estiamted_scale = queue_estimated_scale.get()
                            self.estimated_scale_per_seq[folder] = estiamted_scale
                        t.join(timeout=1)



                while not queue_selected_indexes.empty():
                    folder, selected_indexes = queue_selected_indexes.get()
                    folders.append(folder)
                    folders_name.append('_'.join(list(Path(folder).parts[-2:])) )
                    self.selected_indexes_per_seq[folder] = selected_indexes
                while not queue_visible_view_indexes.empty():
                    folder, visible_view_indexes = queue_visible_view_indexes.get()
                    self.visible_view_indexes_per_seq[folder] = visible_view_indexes
                while not queue_view_indexes_per_point.empty():
                    folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                    self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
                while not queue_clean_point_list.empty():
                    folder, clean_point_list = queue_clean_point_list.get()
                    self.clean_point_list_per_seq[folder] = clean_point_list
                while not queue_intrinsic_matrix.empty():
                    folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                    self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
                while not queue_extrinsics.empty():
                    folder, extrinsics = queue_extrinsics.get()
                    self.extrinsics_per_seq[folder] = extrinsics
                while not queue_projection.empty():
                    folder, projection = queue_projection.get()
                    self.projection_per_seq[folder] = projection
                while not queue_point_cloud.empty():
                    folder, point_cloud = queue_point_cloud.get()
                    self.point_cloud_per_seq[folder] = point_cloud
                print("Pre-processing complete.")



                for (i,folder) in enumerate(folders):
                    precompute_path=self.intermediate_data_root / ("precompute_{}.pkl".format(folders_name[i]))
                    with open(str(precompute_path), "wb") as f:
                        pickle.dump(
                            {"selected_indexes_per_seq": self.selected_indexes_per_seq[folder],
                            "visible_view_indexes_per_seq": self.visible_view_indexes_per_seq[folder],
                            "point_cloud_per_seq": self.point_cloud_per_seq[folder],
                            "intrinsic_matrix_per_seq": self.intrinsic_matrix_per_seq[folder],
                            "view_indexes_per_point_per_seq": self.view_indexes_per_point_per_seq[folder],
                            "extrinsics_per_seq": self.extrinsics_per_seq[folder],
                            "projection_per_seq": self.projection_per_seq[folder],
                            "clean_point_list_per_seq": self.clean_point_list_per_seq[folder],
                            "out_size": self.out_size,
                            "network_downsampling": self.network_downsampling,
                            "inlier_percentage": self.inlier_percentage},
                            f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        if self.phase == "train":
            if self.num_iter is not None:
                return max(self.num_iter, len(self.image_file_names))
            else:
                return len(self.image_file_names)
        elif self.phase == "validation" or self.phase == "onlyval":
            if self.num_iter is not None:
                return 3000
            else:
                return len(self.image_file_names)
        else:
            return len(self.image_file_names)

    def apply_augmentations(self, image1,image2):
        image2_dict = {'image': image2}
        result2 = self.aug_func(**image2_dict)
        return image1, result2['image']

    def apply_augmentations0(self, image1, image2):
        while 1:
            step=0
            error=0
            while 1 :
                mask=np.zeros(image1.shape)
                randomh0=random.sample(range(self.sampling_size+0), self.sampling_size+0)
                randomw0=random.sample(range(self.sampling_size+0), self.sampling_size+0)
                for i in range(self.sampling_size+0):
                    mask[int(randomh0[i]*image1.shape[0]/(self.sampling_size+0)+image1.shape[0]/(self.sampling_size+0)/2),
                        int(randomw0[i]*image1.shape[1]/(self.sampling_size+0)+image1.shape[1]/(self.sampling_size+0)/2),:]=1
                ps1=np.zeros((self.sampling_size+0,2))
                ps1[:,1]=np.where(mask[:,:,0]==1)[0]
                ps1[:,0]=np.where(mask[:,:,0]==1)[1]

                result2=self.aug_func0(image=image2,mask=mask)

                if (len(np.where(result2['mask'][:,:,0]==1)[0])!=self.sampling_size+0) or (len(np.where(result2['mask'][:,:,0]==1)[1])!=self.sampling_size+0):
                    continue
                else:
                    k1=np.where(result2['mask'][:,:,0]==1)[0]
                    k2=np.where(result2['mask'][:,:,0]==1)[1]
                    break
            ps2=np.zeros((self.sampling_size+0,2))
            ps2[:,1]=k1
            ps2[:,0]=k2

            falseerror=[]
            col_num = 0
            p1sort=np.argsort(ps1[:, col_num])
            p2sort=np.argsort(ps2[:, col_num])
            for i in range(self.sampling_size+0):
                if p1sort[i]!=p2sort[i]:
                    error+=1
            if error>self.sampling_size//2:
                continue
            while 1:
                step+=1
                col_num = 0
                p1sort=np.argsort(ps1[:, col_num])
                p2sort=np.argsort(ps2[:, col_num])
                wrongindex=[]
                for i in range(self.sampling_size+0):
                    if (p1sort[i]!=p2sort[i]) and [p1sort[i],p2sort[i]] not in falseerror:
                        wrongindex.append([p1sort[i],p2sort[i]])
                        if np.linalg.norm(ps2[p2sort[i],:]-ps1[p1sort[i],:])<np.linalg.norm(ps2[p2sort[i],:]-ps1[p2sort[i],:]):
                            t=ps2[p1sort[i],:].copy()
                            ps2[p1sort[i],:]=ps2[p2sort[i],:]
                            ps2[p2sort[i],:]=t
                        else:falseerror.append([p1sort[i],p2sort[i]])
                        break
                if wrongindex==[] or step>self.sampling_size:
                    break
            if step<self.sampling_size:
                break

        return image1, result2['image'],ps1,ps2

    def get_pair(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[0:2]
        homo_matrix = None
        while homo_matrix is None:
            homo_matrix = utils.get_perspective_mat(self.aug_params['patch_ratio'], width, height,
                                              self.aug_params['translation'])
            try:
                torch.inverse(torch.from_numpy(homo_matrix))
            except:
                homo_matrix = None
        if self.apply_aug:
            image,img,ps1,ps2= self.apply_augmentations0(image, image)

            warped_image = cv2.warpPerspective(img, homo_matrix, (width, height))
        else:
            warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))


        if self.st == 0:
            if height > width:
                self.config['image_width'], self.config['image_height'] = self.out_size[1], self.out_size[0]
            else:
                self.config['image_width'], self.config['image_height'] = self.out_size[0], self.out_size[1]
        else:
            if height > width:
                self.config['image_height'] = self.out_size[0]
                self.config['image_width'] = int((self.config['image_height'] / height * width) // self.st * self.st)
            else:
                self.config['image_width'] = self.out_size[1]
                self.config['image_height'] = int((self.config['image_width'] / width * height) // self.st * self.st)

        if self.resize:
            orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
            warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
            ps1[:,0]=ps1[:,0]/image.shape[1]*self.config['image_width']
            ps1[:,1]=ps1[:,1]/image.shape[0]*self.config['image_height']
            ps2[:,0]=ps2[:,0]/image.shape[1]*self.config['image_width']
            ps2[:,1]=ps2[:,1]/image.shape[0]*self.config['image_height']

        else:
            orig_resized = image
            warped_resized = warped_image
        if self.apply_aug:
            orig_resized, warped_resized = self.apply_augmentations(orig_resized, warped_resized)
        homo_matrix = utils.scale_homography(homo_matrix, height, width, self.config['image_height'],
                                       self.config['image_width']).astype(np.float32)
        orig_resized =orig_resized.astype(np.float32) / 255.0
        warped_resized = warped_resized.astype(np.float32) / 255.0

        homography = torch.from_numpy(homo_matrix).unsqueeze(0)
        valid_mask_right = utils.compute_valid_mask(orig_resized.shape, homography)
        homography = homography.numpy()[0]
        valid_mask_right=valid_mask_right.numpy()[0]

        training_color_img_1=self.transform({"image": orig_resized})["image"]
        training_color_img_2=self.transform({"image": warped_resized})["image"]
        height, width = training_color_img_1.shape[1:]

        ps2warp=np.zeros((self.sampling_size+0,2),dtype=np.float32)
        for i in range(self.sampling_size+0):
            p0=np.array([[ps2[i,0]],[ps2[i,1]],[1]])
            p1=homography @ p0
            p1=p1*(1/p1[2])
            randomw1=round(p1[0,0])
            randomh1=round(p1[1,0])
            if 0 <= randomw1 < width and 0 <= randomh1 < height and valid_mask_right[randomh1,randomw1]!=0:
                ps2warp[i,:]=[randomw1,randomh1]

        if (np.argwhere(ps2warp[:,0]==0)).size != 0:
            nonzero=np.argwhere(ps2warp[:,0]!=0).reshape(-1)
            ps2warp=ps2warp[nonzero]
            ps1=ps1[nonzero]

        return training_color_img_1, training_color_img_2, homography,valid_mask_right,ps1,ps2warp

    def __getitem__(self, idx):
        if self.phase == 'train' or self.phase == "validation"   :
            p = np.random.random()
            if (p >0.2 or self.phase == "validation")  :
                while True:

                    img_file_name = self.image_file_names[idx % len(self.image_file_names)]
                    folder = img_file_name.parents[1]
                    images_folder = folder / "images"
                    folder_str = str(folder)
                    for i, dataset in enumerate(self.datasets_list):
                        if folder.name in dataset:
                            self.real_flag = i
                            break
                    if self.phase == 'train':                
                        if self.train_phase=='train_real' and self.real_flag<2: # train on real data only 
                            idx = np.random.randint(0, len(self.image_file_names))
                            continue
                        if self.train_phase=='train_synthetic' and self.real_flag>=2: # train on synthetic data only
                            idx = np.random.randint(0, len(self.image_file_names))
                            continue
                    else:
                        if self.real_flag<2:  # validation on real data only
                            idx = np.random.randint(0, len(self.image_file_names))
                            continue

                    precompute_path=self.intermediate_data_root / ("precompute_{}.pkl".format('_'.join(list(Path(folder).parts[-2:])) ))
                    with open(str(precompute_path), "rb") as f:
                        [self.selected_indexes_per_seq,
                        self.visible_view_indexes_per_seq,
                        self.point_cloud_per_seq,
                        self.intrinsic_matrix_per_seq,
                        self.view_indexes_per_point_per_seq,
                        self.extrinsics_per_seq,
                        self.projection_per_seq,
                        self.clean_point_list_per_seq,
                        self.out_size, self.network_downsampling,
                        self.inlier_percentage] = pickle.load(f).values()

                    pos, increment = utils.generating_pos_and_increment(idx=idx,
                                                                        visible_view_indexes=
                                                                        self.visible_view_indexes_per_seq,
                                                                        adjacent_range=self.adjacent_range)

                    pair_indexes = [self.visible_view_indexes_per_seq[pos],
                                    self.visible_view_indexes_per_seq[pos + increment]]

                    camera_intrinsic=self.intrinsic_matrix_per_seq[:,0:3]

                    pair_extrinsics_matrices = [self.extrinsics_per_seq[pos],
                                                self.extrinsics_per_seq[pos + increment]]
                    pair_projection_matrices = [self.projection_per_seq[pos],
                                                self.projection_per_seq[pos + increment]]
                    pair_intrinsic_matrices = self.intrinsic_matrix_per_seq

                    pair_imgs = utils.get_pair_color_imgs(images_folder,pair_indexes,self.out_size)
                    training_color_img_1=self.transform({"image": pair_imgs[0]})["image"]
                    training_color_img_2=self.transform({"image": pair_imgs[1]})["image"]
                    height, width = training_color_img_1.shape[1:]

                    feature_matches = \
                        utils.get_torch_training_data_feature_matching(height=height, width=width,camera_intrinsic=camera_intrinsic,
                                                                    pair_projections=
                                                                    pair_projection_matrices,
                                                                    pair_extrinsics=
                                                                    pair_extrinsics_matrices,
                                                                    pair_indexes=pair_indexes,
                                                                    out_size=self.out_size,
                                                                    point_cloud=self.point_cloud_per_seq,
                                                                    view_indexes_per_point=
                                                                    self.view_indexes_per_point_per_seq,
                                                                    visible_view_indexes=
                                                                    self.visible_view_indexes_per_seq,
                                                                    clean_point_list=
                                                                    self.clean_point_list_per_seq,
                                                                    reprojection_error_threshold=self.reprojection_error_threshold)

                    if feature_matches.shape[0] > 0:
                        sampled_feature_matches_indexes = \
                            np.asarray(
                                np.random.choice(np.arange(feature_matches.shape[0]), size=self.sampling_size),
                                dtype=np.int32).reshape((-1,))
                        sampled_feature_matches = np.asarray(feature_matches[sampled_feature_matches_indexes, :],
                                                            dtype=np.float32).reshape(
                            (self.sampling_size, 4))
                        break
                    else:
                        idx = np.random.randint(0, len(self.image_file_names))
                        continue

                _,height, width = training_color_img_1.shape
                training_heatmaps_1, training_heatmaps_2 = utils.generate_heatmap_from_locations(
                    sampled_feature_matches, height, width, self.heatmap_sigma)


                training_mask_boundary = utils.type_float_and_reshape(np.ones((height,width)),(1,height, width))
                valid_mask_right=training_mask_boundary

                source_feature_2D_locations = sampled_feature_matches[:, :2]
                target_feature_2D_locations = sampled_feature_matches[:, 2:]

                source_feature_1D_locations = np.zeros(
                    (sampled_feature_matches.shape[0], 1), dtype=np.int32)
                target_feature_1D_locations = np.zeros(
                    (sampled_feature_matches.shape[0], 1), dtype=np.int32)

                clipped_source_feature_2D_locations = source_feature_2D_locations
                clipped_source_feature_2D_locations[:, 0] = np.clip(clipped_source_feature_2D_locations[:, 0], a_min=0,
                                                                    a_max=width - 1)
                clipped_source_feature_2D_locations[:, 1] = np.clip(clipped_source_feature_2D_locations[:, 1], a_min=0,
                                                                    a_max=height - 1)

                clipped_target_feature_2D_locations = target_feature_2D_locations
                clipped_target_feature_2D_locations[:, 0] = np.clip(clipped_target_feature_2D_locations[:, 0], a_min=0,
                                                                    a_max=width - 1)
                clipped_target_feature_2D_locations[:, 1] = np.clip(clipped_target_feature_2D_locations[:, 1], a_min=0,
                                                                    a_max=height - 1)

                source_feature_1D_locations[:, 0] = np.round(clipped_source_feature_2D_locations[:, 0]) + \
                                                    np.round(clipped_source_feature_2D_locations[:, 1]) * width
                target_feature_1D_locations[:, 0] = np.round(clipped_target_feature_2D_locations[:, 0]) + \
                                                    np.round(clipped_target_feature_2D_locations[:, 1]) * width

            else:
                while True:

                    img_file_name = self.image_file_names[idx % len(self.image_file_names)]
                    folder = img_file_name.parents[1]
                    images_folder = folder / "images"
                    folder_str = str(folder)
                    for i, dataset in enumerate(self.datasets_list):
                        if folder.name in dataset:
                            self.real_flag = i  # Assign i for the corresponding dataset
                            break

                    if self.train_phase=='train_real' and self.real_flag<2:
                        idx = np.random.randint(0, len(self.image_file_names))
                        continue
                    if self.train_phase=='train_synthetic' and self.real_flag>=2:
                        idx = np.random.randint(0, len(self.image_file_names))
                        continue

                    precompute_path=self.intermediate_data_root / ("precompute_{}.pkl".format('_'.join(list(Path(folder).parts[-2:])) ))
                    with open(str(precompute_path), "rb") as f:
                        [self.selected_indexes_per_seq,
                        self.visible_view_indexes_per_seq,
                        self.point_cloud_per_seq,
                        self.intrinsic_matrix_per_seq,
                        self.view_indexes_per_point_per_seq,
                        self.extrinsics_per_seq,
                        self.projection_per_seq,
                        self.clean_point_list_per_seq,
                        self.out_size, self.network_downsampling,
                        self.inlier_percentage] = pickle.load(f).values()


                    pos, _ = utils.generating_pos_and_increment(idx=idx,
                                                                        visible_view_indexes=
                                                                        self.visible_view_indexes_per_seq,
                                                                        adjacent_range=self.adjacent_range)
                    pair_indexes = self.visible_view_indexes_per_seq[pos]

                    pair_intrinsic_matrices = self.intrinsic_matrix_per_seq
                    f_p= os.path.join(images_folder, "{:05d}".format(pair_indexes))
                    if os.path.exists(f_p + ".png"):
                        file_path = f_p + ".png"
                    elif os.path.exists(f_p + ".jpg"):
                        file_path = f_p + ".jpg"

                    training_color_img_1, training_color_img_2, homography,valid_mask_right,ps1,ps2 = self.get_pair(file_path=file_path)

                    sampled_feature_matches=np.zeros((ps1.shape[0],4),dtype=np.float32)
                    sampled_feature_matches[:,0:2]=ps1
                    sampled_feature_matches[:,2:4]=ps2

                    break


                _,height, width = training_color_img_1.shape
                training_heatmaps_1, training_heatmaps_2 = utils.generate_heatmap_from_locations(
                    sampled_feature_matches, height, width, self.heatmap_sigma)


                training_mask_boundary = utils.type_float_and_reshape(np.ones((height,width)),(1,height, width))
                valid_mask_right=utils.type_float_and_reshape(valid_mask_right,(1,height, width))

                source_feature_2D_locations = sampled_feature_matches[:, :2]
                target_feature_2D_locations = sampled_feature_matches[:, 2:]

                source_feature_1D_locations = np.zeros(
                    (sampled_feature_matches.shape[0], 1), dtype=np.int32)
                target_feature_1D_locations = np.zeros(
                    (sampled_feature_matches.shape[0], 1), dtype=np.int32)

                clipped_source_feature_2D_locations = source_feature_2D_locations
                clipped_source_feature_2D_locations[:, 0] = np.clip(clipped_source_feature_2D_locations[:, 0], a_min=0,
                                                                    a_max=width - 1)
                clipped_source_feature_2D_locations[:, 1] = np.clip(clipped_source_feature_2D_locations[:, 1], a_min=0,
                                                                    a_max=height - 1)

                clipped_target_feature_2D_locations = target_feature_2D_locations
                clipped_target_feature_2D_locations[:, 0] = np.clip(clipped_target_feature_2D_locations[:, 0], a_min=0,
                                                                    a_max=width - 1)
                clipped_target_feature_2D_locations[:, 1] = np.clip(clipped_target_feature_2D_locations[:, 1], a_min=0,
                                                                    a_max=height - 1)

                source_feature_1D_locations[:, 0] = np.round(clipped_source_feature_2D_locations[:, 0]) + \
                                                    np.round(clipped_source_feature_2D_locations[:, 1]) * width
                target_feature_1D_locations[:, 0] = np.round(clipped_target_feature_2D_locations[:, 0]) + \
                                                    np.round(clipped_target_feature_2D_locations[:, 1]) * width

            return [    np.stack([training_color_img_1, training_color_img_2], axis=0),
                        source_feature_1D_locations,
                        target_feature_1D_locations,
                        source_feature_2D_locations,
                        target_feature_2D_locations,
                        training_heatmaps_1,
                        training_heatmaps_2,
                        training_mask_boundary,
                        valid_mask_right,
                        pair_intrinsic_matrices,
                        self.real_flag
                        ]

        elif self.phase == "test":
            img_file_name = self.image_file_names[idx]
            img_file_name=Path(img_file_name)
            folder = img_file_name.parents[1]
            folder_str = str(folder)

            precompute_path=self.intermediate_data_root / ("precompute_{}.pkl".format('_'.join(list(Path(folder).parts[-2:])) ))
            with open(str(precompute_path), "rb") as f:
                [self.selected_indexes_per_seq,
                self.visible_view_indexes_per_seq,
                self.point_cloud_per_seq,
                self.intrinsic_matrix_per_seq,
                self.view_indexes_per_point_per_seq,
                self.extrinsics_per_seq,
                self.projection_per_seq,
                self.clean_point_list_per_seq,
                self.out_size, self.network_downsampling,
                self.inlier_percentage] = pickle.load(f).values()

            camera_intrinsic=self.intrinsic_matrix_per_seq[:,0:3]
            img_list = []
            projection_matrix_list = []
            extrinsic_matrix_list=[]

            for i in range(idx, min(idx + self.adjacent_range[1] + 1, len(self.image_file_names))):
                img = utils.read_color_img(self.image_file_names[i], self.out_size)
                # Normalize
                img_list.append( torch.from_numpy(self.transform({"image":img})["image"]).unsqueeze(dim=0))
                projection_matrix_list.append(self.projection_per_seq[i])
                extrinsic_matrix_list.append(self.extrinsics_per_seq[i])

            folder_name = folder_str.split('/')[-1]
            folder_file_path = self.intermediate_data_root / f"clear_precompute_1_{folder_name}.pkl"
            try:
                with open(str(folder_file_path), "rb") as f:
                    loaded_data = pickle.load(f)
                    feature_matches_list = loaded_data[idx]
            except (IndexError, KeyError, TypeError):
                feature_matches_list = []

            height, width = img_list[0].shape[2:]
            training_mask_boundary = utils.type_float_and_reshape(np.ones((height,width)),(1,height, width))
            return [torch.cat(img_list, dim=0), feature_matches_list, img_to_tensor(training_mask_boundary),folder_str,idx]

        elif self.phase == 'image_loading':
            img_file_name = self.image_file_names[idx]
            folder_str = str(img_file_name.parents[1])

            color_img = utils.read_color_img(img_file_name, self.out_size)
            training_color_img_1=self.transform({"image": color_img})["image"]
            _,height, width = training_color_img_1.shape

            if not os.path.exists(folder_str + '/intrinsic.txt'):
                pair_intrinsic_matrices = 0
            else:
                pair_intrinsic_matrices = np.genfromtxt(folder_str + '/intrinsic.txt', delimiter=",").T

            training_mask_boundary = utils.type_float_and_reshape(np.ones((height,width)),(1,height, width))

            return [training_color_img_1,
                    training_mask_boundary,
                    str(img_file_name), folder_str,pair_intrinsic_matrices]


