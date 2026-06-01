import json
import torch.nn.functional as F
import torch
import random
import scipy.io
import numpy as np
import albumentations as A
import cv2
from skimage import io
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class DataFolder(Dataset):
    def __init__(
            self,
            cfg,
            mode
    ):
        super(DataFolder, self).__init__()

        self.mode = mode
        dataset = cfg.data.name

        self.files = np.load(f'datasets/{dataset}_{mode}_files.npy')

        self.dataset = dataset

        self.transform = A.Compose(
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform]
            + [ToTensorV2()], p=1)
        self.num_mask_per_img = cfg.data.num_mask_per_img
        self.num_neg_prompt = cfg.data.num_neg_prompt

        if 'pannuke' in self.dataset:
            if mode == 'train':
                fid = self.dataset[-3]
                #self.files = [f'datasets/pannuke/Images/{fid}_{0}.png']
            elif mode == 'val':
                fid = self.dataset[-2]
                #self.files = [f'datasets/pannuke/Images/{fid}_{i}.png' for i in range(len(self.files))]
            else:
                fid = self.dataset[-1]

            self.files = [f'datasets/pannuke/Images/{fid}_{i}.png' for i in range(len(self.files))]
            self.types = np.load(f'datasets/pannuke/fold {fid}/images/fold{fid}/types.npy')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        if self.dataset == 'kumar':
            mask_path = f'{img_path[:-4].replace("images", "labels")}.npy'
            sub_paths = mask_path.split('/')
            sub_paths[-2] += '_ins'
            mask_path = '/'.join(sub_paths)
        elif self.dataset == 'cpm17':
            mask_path = f'{img_path[:-4].replace("Images", "Labels")}.mat'
        elif self.dataset == 'monuseg':
            mask_path = f'{img_path[:-4].replace("images", "labels")}.npy'
        else:
            mask_path = f'{img_path[:-4].replace("Images", "Masks")}.npy'

        img, mask = io.imread(img_path)[..., :3], load_maskfile(mask_path)

        

        if self.mode != 'train':
            image_name = img_path.split('/')[-1]
            image = torch.tensor(img.copy())
            res = self.transform(image=img)

            img, mask = res['image'], torch.as_tensor(mask)
            inst_map, type_map = mask[..., 0], mask[..., 1]
            ori_size = inst_map.shape

            img_name = img_path.split('/')[-1]
            if 'pannuke' in self.dataset:
                prompt_points = np.load(f'prompts/{"our_" + self.dataset}/{img_name[:-4]}.npy')
            else:
                prompt_points = np.load(f'prompts/{"our_" + self.dataset}/test/{img_name[:-4]}.npy')
            prompt_points = torch.from_numpy(prompt_points).float()
            prompt_points, prompt_cell_types = prompt_points[..., :2].unsqueeze(1), prompt_points[..., -1]
            prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)

            return img, inst_map, type_map, prompt_points, prompt_labels, prompt_cell_types, ori_size, idx, image, image_name

        two_class_mask = mask[...,0].copy()
        two_class_mask[two_class_mask>1] = 1
        edge = sobel_edge_detection(two_class_mask)
        edge = np.expand_dims(edge, axis=2)
        mask = np.concatenate([mask, edge], axis=-1)


        img_name = img_path.split('/')[-1]
        if 'pannuke' in self.dataset:
            prompt_points_train = np.load(f'prompts/{"our_" + self.dataset}/{img_name[:-4]}.npy')
        else:
            prompt_points_train = np.load(f'prompts/{"our_" + self.dataset}/test/{img_name[:-4]}.npy')
        prompt_points_train = torch.from_numpy(prompt_points_train).float()
        prompt_points_train, prompt_cell_types_train = prompt_points_train[..., :2].unsqueeze(1), prompt_points_train[
            ..., -1]
        point_coord = prompt_points_train.long()
        mask_point = create_mask_with_points(mask.shape[:2],point_coord).unsqueeze(2).numpy()
        mask = np.concatenate([mask,mask_point],axis=-1)

        res = self.transform(image=img, mask=mask)
        img, mask = list(res.values())

        inst_map, type_map, gt_edge, point_map = mask[..., 0], mask[..., 1], mask[...,2], mask[...,3]
        unique_pids = np.unique(inst_map)[1:]  # remove zero

        points_coord = extract_points_with_value_1(point_map).unsqueeze(1)
        normalized_points = points_coord.clone().float()
        normalized_points[:, 0, 0] = 2 * points_coord[:, 0, 0] / (256 - 1) - 1
        normalized_points[:, 0, 1] = 2 * points_coord[:, 0, 1] / (256 - 1) - 1
        ids = F.grid_sample(inst_map.float().unsqueeze(0).unsqueeze(0), normalized_points.unsqueeze(0), mode='bilinear', align_corners=True).long().numpy()
        id_coord = {}
        for i, id in enumerate(ids.squeeze(0).squeeze(0).squeeze(1)):
            if id not in id_coord.keys() and id != 0:
                id_coord[id] = points_coord[i, :, :]
        unique_pids = np.unique(ids)
        if 0 in ids:
            unique_pids = unique_pids[1:]

        cell_num = len(unique_pids)

        # res = self.transform(image=img, mask=mask)
        # img, mask = list(res.values())

        # inst_map, type_map, gt_edge = mask[..., 0], mask[..., 1], mask[...,2]
        # unique_pids = np.unique(inst_map)[1:]  # remove zero

        # cell_num = len(unique_pids)


        if cell_num:
            all_points = []
            cell_types = []

            for pid in unique_pids:
                mask_single_cell = torch.eq(
                    inst_map,
                    pid
                )
                pt = id_coord[pid]

                all_points.append(pt)
                assert type_map[pt[0, 1], pt[0, 0]] > 0
                cell_types.append(type_map[pt[0, 1], pt[0, 0]] - 1)

            all_points = torch.from_numpy(np.concatenate(all_points)).float()

            chosen_pids = np.random.choice(
                unique_pids,
                min(cell_num, self.num_mask_per_img),
                replace=False
            )

            inst_maps = []
            prompt_points = []
            gt_edges = []
            for pid in chosen_pids:
                mask_single_cell = torch.eq(inst_map, pid)
                single_cell_edge = torch.where(mask_single_cell, gt_edge, torch.zeros_like(gt_edge))
                inst_maps.append(mask_single_cell)
                gt_edges.append(single_cell_edge)
                prompt_points.append(id_coord[pid])

            prompt_points = torch.stack(prompt_points, dim=0)
            prompt_labels = torch.ones(prompt_points.shape[:2])
            cell_types = torch.as_tensor(cell_types)

            inst_map = torch.stack(inst_maps, dim=0)
            gt_edge = torch.stack(gt_edges, dim=0)

            if self.num_neg_prompt:
                global_indices = [np.where(unique_pids == pid)[0][0] for pid in chosen_pids]

                prompt_points, prompt_labels = add_k_nearest_neg_prompt(
                    prompt_points,
                    global_indices,
                    all_points,
                    k=self.num_neg_prompt
                )
        else:
            prompt_points = torch.empty(0, (self.num_neg_prompt + 1), 2)
            prompt_labels = torch.empty(0, (self.num_neg_prompt + 1))
            all_points = torch.empty(0, 2)
            inst_map = torch.empty(0, 256, 256)
            gt_edge = torch.empty(0, 256, 256)
            cell_types = torch.empty(0)

        return img, inst_map.long(), prompt_points, prompt_labels, cell_types, all_points, gt_edge.long()


def sobel_edge_detection(mask):
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif len(mask.shape) != 2:
        raise ValueError("输入掩码必须是二维的或形状为 (1, H, W)")

    mask = torch.tensor(mask, dtype=torch.float32)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    edge_x = F.conv2d(mask.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
    edge_y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)

    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    edge = edge / edge.max()

    return edge.squeeze().numpy()

def create_mask_with_points(mask_size, points):
    mask = torch.zeros(mask_size, dtype=torch.float32)

    points = points.squeeze(1)  # 转换为 (n, 2)

    for point in points:
        x, y = point
        if 0 <= x < mask_size[1] and 0 <= y < mask_size[0]:
            mask[int(y), int(x)] = 1.0

    return mask


def extract_points_with_value_1(mask):
    y_coords, x_coords = torch.where(mask == 1)

    points = torch.stack((x_coords, y_coords), dim=1)

    return points

def load_maskfile(mask_path: str):
    if 'pannuke' in mask_path:
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        type_map = mask[()]["type_map"].astype(np.int32)

    elif 'cpm17' in mask_path:
        inst_map = scipy.io.loadmat(mask_path)['inst_map']
        type_map = (inst_map.copy() > 0).astype(float)

    else:
        inst_map = np.load(mask_path)
        type_map = (inst_map.copy() > 0).astype(float)

    mask = np.stack([inst_map, type_map], axis=-1)
    return mask


def add_k_nearest_neg_prompt(
        prompt_points,
        global_indices,
        all_points,
        k: int = 1
):
    if len(prompt_points) == 1:
        prompt_points = torch.cat([prompt_points, torch.zeros(1, k, 2)], dim=1)
        prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)
        prompt_labels[0, 1] = -1
    else:
        all_points = all_points.view(-1, 2)
        dis = torch.cdist(all_points, all_points, p=2.0)
        dis = dis.fill_diagonal_(np.inf)

        available_num = min(k, len(prompt_points) - 1)
        neg_prompt_points = all_points[
                            torch.topk(dis[global_indices], available_num, dim=1, largest=False).indices, :
                            ]
        prompt_points = torch.cat(
            [prompt_points, neg_prompt_points, torch.zeros(len(prompt_points), k - available_num, 2)],
            dim=1
        )

        prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)
        prompt_labels[:, 1:available_num + 1] = 0
        prompt_labels[:, available_num + 1:] = -1

    return prompt_points, prompt_labels
