from typing import Optional, Callable, Union, List
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image


class CIFAR10DataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = '~/data/',
                 img_size: tuple = (32, 32),
                 batch_size: int = 64,
                 valid_size: float = 0.2,
                 num_workers: int = 10,
                 seed: int = 10,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 transforms_type: str = 'default',
                 train_transforms = None,
                 valid_transforms = None,
                 test_transforms = None) -> None:
        super().__init__()

        if transforms_type not in ['default', 'custom', 'anomaly']:
            raise ValueError(f'`transforms_type` value is not supported. Got "{transforms_type}" value.')

        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transforms_type = transforms_type

        self.norm_mean = [0.49139968, 0.48215841, 0.44653091]
        self.norm_std = [0.24703223, 0.24348513, 0.26158784]
        self.num_classes = 10

        if self.transforms_type == 'default':
            self.train_transforms = Transforms(self._get_default_transforms(split='train'))
            self.valid_transforms = Transforms(self._get_default_transforms(split='valid'))
            self.test_transforms = Transforms(self._get_default_transforms(split='test'))
        elif self.transforms_type == 'custom':
            self.train_transforms = train_transforms
            self.valid_transforms = valid_transforms
            self.test_transforms = test_transforms
        elif self.transforms_type == 'anomaly':
            self.train_transforms = Transforms(self._get_anomaly_transforms())
            self.valid_transforms = Transforms(self._get_anomaly_transforms())
            self.test_transforms = Transforms(self._get_anomaly_transforms())
        else:
            raise NotImplementedError('transforms_type {} not implemented.'.format(transforms_type))

    def prepare_data(self) -> None:
        CIFAR10(root=self.data_dir, train=True, download=True) # Train split
        CIFAR10(root=self.data_dir, train=False, download=True) # Test split

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ["fit", "validate", "test", "predict"]:
            raise ValueError(f' stage value is not supported. Got "{stage}" value.')
        
        if stage == 'fit' or stage == 'validate':
            self.ds_cifar10_train_valid = CIFAR10(root=self.data_dir,
                                                  train=True,
                                                  transform=self.train_transforms)
            self.num_classes = len(self.ds_cifar10_train_valid.classes)

            self.ds_cifar10_train, self.ds_cifar10_valid = self._split_dataset(self.ds_cifar10_train_valid)

            # self.ds_cifar10_train = self._split_dataset(self.ds_cifar10_train_valid,
            #                                             train=True)
            # self.ds_cifar10_valid = self._split_dataset(self.ds_cifar10_train_valid,
            #                                             train=False)
        # elif stage == 'validate':
        #     self.ds_cifar10_train_valid = CIFAR10(root=self.data_dir,
        #                                           train=True,
        #                                           transform=self.valid_transforms)
        #     self.num_classes = len(self.ds_cifar10_train_valid.classes)
        #     self.ds_cifar10_valid = self._split_dataset(self.ds_cifar10_train_valid,
        #                                                 train=False)
        elif stage == 'test':
            self.ds_cifar10_test = CIFAR10(root=self.data_dir,
                                           train=False,
                                           transform=None)
            self.num_classes = len(self.ds_cifar10_test.classes)
        else: # predict stage
            pass

    def _split_dataset(self, dataset: Dataset) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset,
                                                  splits,
                                                  generator=torch.Generator().manual_seed(self.seed))
        
        return dataset_train, dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.valid_size, int):
            train_len = len_dataset - self.valid_size
            splits = [train_len, self.valid_size]
        elif isinstance(self.valid_size, float):
            val_len = int(self.valid_size * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported self.valid_size type {type(self.valid_size)}")

        return splits
    
    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.ds_cifar10_train,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=self.persistent_workers)

        return train_loader
    
    def val_dataloader(self) -> DataLoader:
        valid_loader = DataLoader(self.ds_cifar10_valid,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=self.persistent_workers)
        return valid_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(self.ds_cifar10_test,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 persistent_workers=self.persistent_workers)
        return test_loader

    def _get_default_transforms(self, split):
        """
        Get images transforms for data augmentation\n
        By default, Albumentations library is used for data augmentation
        https://albumentations.ai/docs/examples/example/

        :param custom_transforms: Custom image data transforms
        :type custom_transforms: Any, torchvision.transforms
        :return: Image data transforms
        :rtype: Any, albumentations.transform, torchvision.transforms
        """
        if split not in ["train", "valid", "test"]:
            raise ValueError(f' `stage` value is not supported. Got "{split}" value.')
        
        if split == 'train':
            return A.Compose(
                [
                    A.Resize(self.img_size[0],
                             self.img_size[1],
                             p=1.0),
                    A.OneOf([
                             A.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4,
                                           hue=0.0),
                             A.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.50,
                                                rotate_limit=30),
                    ], p=0.5),
                    A.Normalize(mean=self.norm_mean,
                                std=self.norm_std),
                    ToTensorV2()
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size[0],
                             self.img_size[1],
                             p=1.0),
                    A.Normalize(mean=self.norm_mean,
                                std=self.norm_std),
                    ToTensorV2()
                ]
            )

    def _get_anomaly_transforms(self) -> Callable:
        anomaly_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1], p=1),
                A.OneOf([
                    # A.MotionBlur(blur_limit=16, p=1.0),
                    A.RandomFog(fog_coef_lower=0.7,
                                fog_coef_upper=0.9,
                                alpha_coef=0.8,
                                p=1.0),
                    A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.7, 0.7),
                                     src_radius=int(self.img_size[1] * 0.8),
                                     num_flare_circles_lower=8,
                                     num_flare_circles_upper=12,
                                     angle_lower=0.5,
                                     p=1.0),
                    A.RandomSnow(brightness_coeff=2.5,
                                 snow_point_lower=0.6,
                                 snow_point_upper=0.8,
                                 p=1.0)
                ], p=1.0),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )
        return anomaly_transforms
    
    def get_num_classes(self) -> int:
        return self.num_classes

    def unprocess_image(self, im, plot=False):
        im = im.squeeze().numpy().transpose((1, 2, 0))
        im = self.norm_std * im + self.norm_mean
        im = np.clip(im, 0, 1)
        im = im * 255
        im = Image.fromarray(im.astype(np.uint8))

        if plot:
            plt.rcParams['figure.figsize'] = [2.54/2.54, 2.54/2.54]
            plt.imshow(im)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            return im

# Helper class to enable albumentations transformations
class Transforms:
    """
    Transforms (dummy) Class to Apply Albumanetations transforms to
    PyTorch ImageFolder dataset class\n
    See:
        https://albumentations.ai/docs/examples/example/
        https://stackoverflow.com/questions/69151052/using-imagefolder-with-albumentations-in-pytorch
        https://github.com/albumentations-team/albumentations/issues/1010
    """
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]

def get_cifar10_input_transformations(cifar10_normalize_inputs: bool,
                                      img_size: int,
                                      data_augmentations: str,
                                      anomalies: bool):
    if cifar10_normalize_inputs:
        if data_augmentations == 'extra':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.247, 0.243, 0.261))
                ]
            )
        elif data_augmentations == 'basic':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.247, 0.243, 0.261)),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.247, 0.243, 0.261)),
                ]
            )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.247, 0.243, 0.261)),
            ]
        )
        if anomalies:
            anomaly_transforms = A.Compose(
                [
                    A.Resize(img_size, img_size, p=1),
                    A.OneOf([
                        # A.MotionBlur(blur_limit=16, p=1.0),
                        A.RandomFog(fog_coef_lower=0.3,  # Orig 0.7
                                    fog_coef_upper=0.5,  # Orig 0.9
                                    alpha_coef=0.4,  # Orig 0.8
                                    p=1.0),
                        A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.7, 0.7),
                                         src_radius=int(img_size * 0.4),  # Orig 0.8
                                         num_flare_circles_lower=4,  # Orig 8
                                         num_flare_circles_upper=8,  # Orig 12
                                         angle_lower=0.4,
                                         p=1.0),
                        A.RandomSnow(brightness_coeff=1.3,  # Orig 2.5
                                     snow_point_lower=0.1,  # Orig 0.6
                                     snow_point_upper=0.3,  # Orig 0.8
                                     p=1.0)
                    ], p=1.0),
                    A.Normalize(mean=np.array([125.3, 123.0, 113.9]), std=np.array([63.0, 62.1, 66.7])),
                    ToTensorV2()
                ]
            )
            return Transforms(anomaly_transforms), Transforms(anomaly_transforms)
    # No cifar10 normalization
    else:
        if data_augmentations == 'extra':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif data_augmentations == 'basic':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.ToTensor(),
            ]
        )
        if anomalies:
            anomaly_transforms = A.Compose(
                [
                    A.Resize(img_size, img_size, p=1),
                    A.OneOf([
                        # A.MotionBlur(blur_limit=16, p=1.0),
                        A.RandomFog(fog_coef_lower=0.7,
                                    fog_coef_upper=0.9,
                                    alpha_coef=0.8,
                                    p=1.0),
                        A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.7, 0.7),
                                         src_radius=int(img_size * 0.8),
                                         num_flare_circles_lower=8,
                                         num_flare_circles_upper=12,
                                         angle_lower=0.5,
                                         p=1.0),
                        A.RandomSnow(brightness_coeff=2.5,
                                     snow_point_lower=0.6,
                                     snow_point_upper=0.8,
                                     p=1.0)
                    ], p=1.0),
                    ToTensorV2()
                ]
            )
            return Transforms(anomaly_transforms), Transforms(anomaly_transforms)

    return train_transforms, test_transforms
