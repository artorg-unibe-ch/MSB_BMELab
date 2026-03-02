import itk
import numpy as np

# flake8: noqa: E501


def crop_bbox(img_np, padding=10):
    """
    Crop a 3D numpy array to the bounding box of non-zero elements with optional padding.
    """
    # Find the coordinates of non-zero elements
    non_zero_coords = np.argwhere(img_np)

    # Determine the bounding box
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1  # +1 to include the max index

    # Apply padding and ensure we don't go out of bounds
    min_coords_pad = np.maximum(min_coords - padding, 0)
    max_coords_pad = np.minimum(max_coords + padding, img_np.shape)

    # Crop the image
    cropped_img = img_np[
        min_coords[0] : max_coords[0],
        min_coords_pad[1] : max_coords_pad[1],
        min_coords_pad[2] : max_coords_pad[2],
    ]

    return cropped_img, (min_coords, max_coords)


def gaussian_smoothing_itk(img_itk, sigma=1.0):
    """
    Apply Gaussian smoothing to an ITK image
    """
    sigma = 1.0 * img_itk.GetSpacing()[0]
    seg_smooth = itk.smoothing_recursive_gaussian_image_filter(img_itk, sigma=sigma)
    return seg_smooth


def otsu_threshold_itk_with_threshold(img_itk):
    """
    Otsu thresholding that returns both binary image and threshold value
    """
    # Cast to float
    float_img = itk.cast_image_filter(
        img_itk, ttype=(type(img_itk), itk.Image[itk.F, 3])
    )

    # Apply Otsu - need to use the filter object to get threshold
    dimension = img_itk.GetImageDimension()
    InputImageType = itk.Image[itk.F, dimension]
    OutputImageType = itk.Image[itk.UC, dimension]

    otsu_filter = itk.OtsuThresholdImageFilter[InputImageType, OutputImageType].New()
    otsu_filter.SetInput(float_img)
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    otsu_filter.Update()

    # Get the threshold value
    threshold = otsu_filter.GetThreshold()

    # Convert to numpy
    binary = itk.array_from_image(otsu_filter.GetOutput())

    return binary, threshold


def visualize_otsu_threshold(img_itk):
    """
    Visualize original image, histogram with threshold, and binary result
    """
    # Apply Otsu and get threshold
    binary, threshold = otsu_threshold_itk_with_threshold(img_itk)

    return binary, threshold
