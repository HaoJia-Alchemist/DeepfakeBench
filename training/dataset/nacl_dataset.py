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
import albumentations as A
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import cv2
from dataset.albu import IsotropicResize, NoiseAugmentation


class NaClDataset(DeepfakeAbstractBaseDataset):

    def __init__(self, config, mode='train'):
        super().__init__(config, mode)
        self.noise_aug = NoiseAugmentation()
        self.fake_num = sum(self.data_dict['label'])
        self.real_num = len(self.data_dict['label']) - self.fake_num
        self.reversed_label_image = []

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.OneOf([A.GaussNoise(p=0.5),
                     A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'],
                                    p=self.config['data_aug']['blur_prob'])]
                    , p=0.5),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ],
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

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
        # Do Noise aug
        if self.config['use_data_augmentation'] and random.random() < self.config['data_aug']['noise_aug_prob']:
            if label == 0:
                # print("real to fake")
                image, mask = self.noise_aug.noise_aug(image, landmarks)
                if not self.config['with_mask']:
                    mask = None
                self.reversed_label_image.append(image_path)
                label = 1
            elif label == 1 and random.random() < self.real_num / self.fake_num and len(self.reversed_label_image) > 0:
                # print("fake to real")
                image_path = self.reversed_label_image.pop()
                label = 0
                # Get the mask and landmark paths
                mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
                landmark_path = image_path.replace('frames', 'landmarks').replace('.png',
                                                                                  '.npy')  # Use .npy for landmark

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

        # Do transforms
        if self.config['use_data_augmentation']:
            try:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
            except Exception as e:
                print(landmark_path, f"{e.args}")
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
        return image_trans, label, landmarks_trans, mask_trans

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
        images, labels, landmarks, masks = zip(*batch)

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

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict


if __name__ == "__main__":
    with open('/home/jh/disk/workspace/DeepfakeBench/training/config/detector/rgbmsnlc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = NaClDataset(
        config=config,
        mode='train',
    )
    from tqdm import tqdm

    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=config['workers'],
            collate_fn=train_set.collate_fn,
        )
    for iteration, batch in enumerate(tqdm(train_data_loader)):

        # diff_img = cv2.absdiff(img, dst)
        print(iteration)
        # print(f"real num:{len(batch['label'])-sum(batch['label'])}, fake num:{sum(batch['label'])}")
        ...
        # if iteration > 10:
        #     break
