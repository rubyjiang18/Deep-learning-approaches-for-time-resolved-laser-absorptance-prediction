''''
Reference https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
Note: images and masks size is exactly 572 * 572
'''

from torchvision import transforms

import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5), #only horizontal to preserve orientation
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0.1, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=572, min_width=572, always_apply=True, border_mode=0),
        albu.GaussNoise(p=0.3),
        albu.RandomBrightnessContrast(p=0.3),
        albu.Perspective(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.3,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1, alpha=(0.2, 0.5)),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),
    ]

    return albu.Compose(train_transform)

# normalize same as imagenet
preprocess =  transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])