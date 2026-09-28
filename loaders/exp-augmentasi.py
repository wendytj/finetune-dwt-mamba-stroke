import torchvision.transforms.v2 as v2
import torch

num_channels = 0

# variabel kontrol yang paling awal
gpu_train_transform = v2.Compose([
    v2.Lambda(
        lambda x: x.repeat(1, 3, 1, 1)
        if (x.ndim == 4 and x.shape[1] == 1 and num_channels == 3)
        else x
    ),
    v2.ToDtype(torch.float32, scale=False), 
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(
        degrees=5,  # type: ignore
        translate=(0.03, 0.03), 
        scale=(0.95, 1.05),
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0
    ),
    v2.RandomApply([
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
    ], p=0.3),
])

# matikan horizontal flip
gpu_train_transform_trial2 = v2.Compose([
    v2.Lambda(
        lambda x: x.repeat(1, 3, 1, 1)
        if (x.ndim == 4 and x.shape[1] == 1 and num_channels == 3)
        else x
    ),
    v2.ToDtype(torch.float32, scale=False), 
    v2.RandomAffine(
        degrees=5,  # type: ignore
        translate=(0.03, 0.03), 
        scale=(0.95, 1.05),
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0
    ),
    v2.RandomApply([
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
    ], p=0.3),
])

# matikan horizontal flip dan gaussian blur
gpu_train_transform_trial3 = v2.Compose([
    v2.Lambda(
        lambda x: x.repeat(1, 3, 1, 1)
        if (x.ndim == 4 and x.shape[1] == 1 and num_channels == 3)
        else x
    ),
    v2.ToDtype(torch.float32, scale=False), 
    v2.RandomHorizontalFlip(p=0.0),  # Dimatikan
    v2.RandomAffine(
        degrees=5,  # type: ignore
        translate=(0.03, 0.03), 
        scale=(0.95, 1.05),
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0
    ),
])