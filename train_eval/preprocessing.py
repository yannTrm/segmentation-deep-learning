import albumentations as A


def get_portrait_validation_augmentation():
    """Add paddings to make portrait image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(
            min_height=1088,  # 1080 + 8 (multiple of 32)
            min_width=608,   
            always_apply=True,
            border_mode=0,    
        ),
    ]
    return A.Compose(test_transform)


def get_landscape_augmentation():
    transform = [
        A.Resize(height=1080, width=1920, always_apply=True),  
        A.PadIfNeeded(min_height=1080, min_width=1920, always_apply=True),  
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(transform)

def get_landscape_validation_augmentation():
    """Add paddings to make landscape image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(
            min_height=1088,  
            min_width=1920,   
            always_apply=True,
            border_mode=0, 
        ),
    ]
    return A.Compose(test_transform)