import albumentations as albu
# from albumentations.augmentations.transforms import ColorJitter
from albumentations.pytorch.transforms import ToTensorV2
# from torchvision.transforms import AutoAugment


def compose(transforms_to_compose):
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


def hard_transforms():
    result = [
        albu.CoarseDropout(
            max_height=32,
            max_width=32,
        ),
        albu.OneOf(
            [
            albu.RandomBrightnessContrast(brightness_limit=0.07),
            albu.GridDistortion(distort_limit=0.1),
            albu.HueSaturationValue(hue_shift_limit=5),
            albu.Blur(blur_limit=2),
            albu.GaussNoise(),
            albu.ColorJitter(),
            ] , p=0.3
        ),
    ]
    return result


def hard_side_transforms():
    result = hard_transforms()
    result.extend([
        albu.HorizontalFlip(), 
        # albu.RandomCrop(int(360*0.75), int(600*0.75))
        ])
    return result


def hard_top_transforms():
    result = hard_transforms()
    result.extend([
        albu.HorizontalFlip(), 
        albu.VerticalFlip(p=0.5), 
        # albu.RandomCrop(int(360*0.75), int(600*0.75))
        ]
    )
    return result


# def resize_transforms(image_size=224):
#     pre_size = int(image_size * 1.5)

#     random_crop = albu.Compose(
#         [
#             albu.SmallestMaxSize(pre_size, p=1),
#             albu.RandomCrop(image_size, image_size, p=1),
#         ]
#     )
#     rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])
#     result = [albu.OneOf([random_crop, rescale], p=1)]

#     return result


def post_transforms():
    return [
        # AutoAugment(),
        albu.Normalize(),
        ToTensorV2(),
    ]


def get_train_top_transforms():
    return compose([hard_top_transforms(), post_transforms()])


def get_train_side_transforms():
    return compose([hard_side_transforms(), post_transforms()])


# def get_show_transforms():
#     return compose([hard_transforms()])


def get_test_transforms():
    return compose([post_transforms()])
