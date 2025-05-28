import albumentations as A

def _calculate_scaled_size(original_size, scale_factor=1.0):
    """Calcule la taille après scaling en s'assurant qu'elle est divisible par 32"""
    scaled_h = int(original_size[0] * scale_factor)
    scaled_w = int(original_size[1] * scale_factor)
    scaled_h = (scaled_h // 32) * 32
    scaled_w = (scaled_w // 32) * 32
    return scaled_h, scaled_w


def get_portrait_augmentation(scale_factor=1.0, grayscale=False):
    original_size = (1080, 608)  # (height, width)
    scaled_h, scaled_w = _calculate_scaled_size(original_size, scale_factor)
    
    transform = [
        A.ToGray(p=1.0) if grayscale else A.NoOp(),
        A.Resize(height=scaled_h, width=scaled_w, always_apply=True),
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
        A.PadIfNeeded(min_height=scaled_h, min_width=scaled_w, always_apply=True),
    ]
    return A.Compose(transform)

def get_portrait_validation_augmentation(scale_factor=1.0, grayscale=False):
    original_size = (1080, 608)  # (height, width)
    scaled_h, scaled_w = _calculate_scaled_size(original_size, scale_factor)
    scaled_h = ((scaled_h + 33) // 32) * 32
    scaled_w = ((scaled_w + 33) // 32) * 32
    
    test_transform = [
        A.ToGray(p=1.0) if grayscale else A.NoOp(),
        A.Resize(height=scaled_h, width=scaled_w, always_apply=True),
    ]
    return A.Compose(test_transform)

def get_landscape_augmentation(scale_factor=1.0):
    original_size = (1080, 1920)  # (height, width)
    scaled_h, scaled_w = _calculate_scaled_size(original_size, scale_factor)
    
    transform = [
        A.Resize(height=scaled_h, width=scaled_w, always_apply=True),
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
        A.PadIfNeeded(min_height=scaled_h, min_width=scaled_w, always_apply=True, border_mode=0),

    ]
    return A.Compose(transform)

def get_landscape_validation_augmentation(scale_factor=1.0):
    original_size = (1080, 1920)  # (height, width)
    scaled_h, scaled_w = _calculate_scaled_size(original_size, scale_factor)
    scaled_h = ((scaled_h + 33) // 32) * 32
    scaled_w = ((scaled_w + 33) // 32) * 32
    
    test_transform = [
    A.Resize(height=scaled_h, width=scaled_w, always_apply=True),
    ]
    return A.Compose(test_transform)
