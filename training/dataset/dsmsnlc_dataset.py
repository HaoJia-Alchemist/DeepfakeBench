'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''
from copy import deepcopy

import torch
import random
import numpy as np
import yaml

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import cv2


class DSMSNLCDataset(DeepfakeAbstractBaseDataset):
    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        # Get the mask and landmark paths
        mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed)
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None
        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
        else:
            landmarks = None
        if self.config['with_noise']:
            nlm_image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
            noise = cv2.absdiff(image, nlm_image)
        else:
            noise = None
        # Do transforms
        if self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
        if self.config['with_noise']:
            noise = self.to_tensor(noise)
        return image_trans, label, landmarks_trans, mask_trans, noise

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks, noises = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        if noises[0] is not None:
            noises = torch.stack(noises, dim=0)
        else:
            noises = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['noise'] = noises
        return data_dict

if __name__ == "__main__":
    with open('/home/jh/disk/workspace/DeepfakeBench/training/config/detector/dsmsnlc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # train_set = DeepfakeAbstractBaseDataset(
    #             config = config,
    #             mode = 'train',
    #         )
    train_set = DSMSNLCDataset(
        config=config,
        mode='train',
    )
    from tqdm import tqdm
    for iteration, filePath in enumerate(tqdm(train_set.image_list)):
        img = cv2.imread(filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        # diff_img = cv2.absdiff(img, dst)
        print(iteration)
        ...
        # if iteration > 10:
        #     break