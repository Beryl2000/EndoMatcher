import cv2
import torch
import numpy as np
import tqdm
from pathlib import Path
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from scipy.ndimage import binary_fill_holes, binary_erosion

import utils
from dpt.models import EndoMacher


def load_model(trained_model_path, device_ids, pre_trained=False):
    model = EndoMacher(
        backbone="vitb_rn50_384",
        non_negative=True,
        pretrainedif=pre_trained,
        enable_attention_hooks=False,
    )
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    if Path(trained_model_path).exists():
        print(f"Loading model from {trained_model_path}...")
        pre_trained_state = torch.load(str(trained_model_path), map_location='cuda:{}'.format(device_ids[0]))
        model_state = model.state_dict()
        trained_model_state = {k: v for k, v in pre_trained_state["model"].items() if k in model_state}
        model_state.update(trained_model_state)
        model.load_state_dict(model_state)
    else:
        raise FileNotFoundError(f"No trained model found at {trained_model_path}")
    return model


def run_feature_matching_pipeline(
    trained_model_path,
    sequence_root,
    out_path,
    input_size,
    batch_size,
    pre_trained,
    rrange,
    cross_check_distance,
    max_feature_detection,
    octave_layers,
    contrast_threshold,
    edge_threshold,
    sigma,
    device_ids
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_root = Path(sequence_root)
    trained_model_path = Path(trained_model_path)
    out_dir = sequence_root / out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: SIFT detector
    sift = cv2.SIFT_create(nfeatures=int(max_feature_detection),nOctaveLayers=int(octave_layers),
                           contrastThreshold=float(contrast_threshold),edgeThreshold=float(edge_threshold),sigma=float(sigma))
    
    # Step 2: Data loading
    colors_list, boundary, _ = utils.gather_feature_matching_data_new_cross(
        sub_folder=sequence_root,
        data_root=sequence_root,
        input_size=input_size,
        network_downsampling=64,
        load_intermediate_data=True,
        precompute_root=sequence_root / "precompute_mix",
        batch_size=batch_size,
        device=device_ids[0]
    )

    # Step 3: Load model
    model = load_model(trained_model_path, device_ids, pre_trained)

    # Step 4: Keypoint extraction
    kernel = np.ones((5, 5), np.uint8)
    boundary = cv2.erode(boundary, kernel, iterations=3)

    print("Extracting keypoints...")
    _, kps_1D_list, kps_2D_list, _ = utils.extract_keypoints(
        sift, colors_list, boundary, input_size[0], input_size[1]
    )

    frame_count = len(colors_list)
    tq = tqdm.tqdm(total=(frame_count - 1 - rrange) * rrange + (1 + rrange) * rrange // 2)

    # Step 5: Matching loop
    for i in range(frame_count):
        color_1 = colors_list[i]
        kps_1D_1 = kps_1D_list[i]
        kps_2D_1 = kps_2D_list[i]

        np.random.seed(10086)

        for j in range(i + 1, min(i + rrange + 1, frame_count)):
            color_2 = colors_list[j]

            input_tensor = torch.from_numpy(np.stack([color_1, color_2], axis=0)).unsqueeze(0).cuda(device=device_ids[0])
            _, _, _, _, _, _, feat_1, feat_2 = model(input_tensor)
            feat_1 = feat_1[0]
            feat_2 = feat_2[0]

            match_result = utils.feature_matching_single_generation(
                feature_map_1=feat_1,
                feature_map_2=feat_2,
                kps_1D_1=kps_1D_1,
                cross_check_distance=cross_check_distance,
                device=device_ids[0]
            )

            if match_result is None:
                tq.update(1)
                continue

            src_idx, tgt_locs = match_result
            src_locs = kps_2D_1[src_idx].reshape((-1, 2))

            img0 = (255 * ((color_1 - color_1.min()) / (color_1.max() - color_1.min()))).astype(np.uint8).transpose(1, 2, 0)
            img1 = (255 * ((color_2 - color_2.min()) / (color_2.max() - color_2.min()))).astype(np.uint8).transpose(1, 2, 0)

            # Foreground masking
            gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray0, 20, 255, cv2.THRESH_BINARY_INV)
            mask = binary_fill_holes(mask == 0).astype(np.uint8)
            mask = binary_erosion(mask, np.ones((30, 30), dtype=bool))

            valid_idx = (
                (src_locs[:, 0] >= 0) & (src_locs[:, 0] < mask.shape[1]) &
                (src_locs[:, 1] >= 0) & (src_locs[:, 1] < mask.shape[0]) &
                (tgt_locs[:, 0] >= 0) & (tgt_locs[:, 0] < mask.shape[1]) &
                (tgt_locs[:, 1] >= 0) & (tgt_locs[:, 1] < mask.shape[0])
            )

            valid_idx = valid_idx & \
                        (mask[src_locs[:, 1].astype(int), src_locs[:, 0].astype(int)] > 0) & \
                        (mask[tgt_locs[:, 1].astype(int), tgt_locs[:, 0].astype(int)] > 0)

            src_valid = src_locs[valid_idx]
            tgt_valid = tgt_locs[valid_idx]

            if len(src_valid) >= 8:
                model_r, inliers = ransac(
                    (src_valid, tgt_valid),
                    FundamentalMatrixTransform,
                    min_samples=8,
                    residual_threshold=1,
                    max_trials=1000
                )

                inliers = inliers if inliers is not None else np.array([])
                colors = np.zeros((np.sum(inliers), 4))
                colors[:] = [0.0, 0.5, 0.0, 0.3]
                text = [f'{np.sum(inliers)}/{len(src_valid)}']

                utils.make_matching_figure(img0, img1, src_valid[inliers], tgt_valid[inliers], color=colors,
                                     text=text, path=str(out_dir / f"i{i}_j{j}.png"))

                # Save 10 sample matches
                n_pick = min(10, np.sum(inliers))
                if n_pick > 0:
                    idx = np.random.choice(np.sum(inliers), n_pick, replace=False)
                    sample_src = src_valid[inliers][idx]
                    sample_tgt = tgt_valid[inliers][idx]
                    colors = np.zeros((n_pick, 4))
                    colors[:] = [0.0, 0.5, 0.0, 1.0]
                    utils.make_matching_figure(img0, img1, sample_src, sample_tgt,
                                         color=colors, text=[f'Sample: {n_pick}'],
                                         path=str(out_dir / f"i{i}_j{j}_sample10.png"))

            tq.update(1)
