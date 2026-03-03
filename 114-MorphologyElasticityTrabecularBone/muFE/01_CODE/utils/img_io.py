from pathlib import Path

import itk
import numpy as np

# flake8: noqa: E501


def read_image(file_path: Path):
    """
    Reads an image from the specified file path using ITK and returns it as a NumPy array.

    Parameters:
        file_path (str or Path): The path to the image file.
    Returns:
        np.ndarray: The image data as a NumPy array.
    """
    suffix = file_path.suffix.lower()
    imageio_map = {
        ".mha": itk.MetaImageIO.New(),
        ".mhd": itk.MetaImageIO.New(),
        ".nii": itk.NiftiImageIO.New(),
        ".nrrd": itk.NrrdImageIO.New(),
    }
    imageio = imageio_map.get(suffix)
    itk_image = itk.imread(str(file_path), imageio=imageio)  # None = auto (fallback)
    np_image = itk.array_from_image(itk_image)
    return itk_image, np_image


def read_mha(file_path: Path):
    """
    Created because ScancoImageIO incorrectly returns True for MHA files -> sniffing logic too permissive
    """
    imageio = itk.MetaImageIO.New()
    itk_image = itk.imread(str(file_path), imageio=imageio)
    np_image = itk.array_from_image(itk_image)
    return itk_image, np_image


def write_image(img_np: np.ndarray, output_path: Path):
    """
    Writes an image to the specified file path.
    Supports ITK formats (.mha, .mhd, .nii, .nrrd, .dcm, .png, .tiff, etc.) and NumPy (.npy).

    Parameters:
        img_np (np.ndarray): The numpy array to be written as an image.
        output_path (str or Path): The path where the image will be saved.
    """
    # Handle boolean arrays (ITK doesn't support bool type)
    if img_np.dtype == np.bool_ or img_np.dtype == bool:
        img_np = img_np.astype(np.uint8)

    output_path = Path(output_path)

    # Handle NumPy format separately
    if output_path.suffix.lower() == ".npy":
        np.save(output_path, img_np)
    else:
        try:
            itk_image = itk.image_from_array(img_np)
            itk.imwrite(itk_image, str(output_path))
        except Exception as e:
            raise NotImplementedError(
                f"Failed to write image in format '{output_path.suffix}'. "
                f"Error: {str(e)}"
            )

    return None


def read_image_metadata(file_path: Path):
    file_path_str = str(file_path)

    # Bypass buggy ImageIOFactory auto-detection, use MetaImageIO directly for .mha/.mhd
    suffix = Path(file_path).suffix.lower()
    imageio_map = {
        ".mha": itk.MetaImageIO,
        ".mhd": itk.MetaImageIO,
        ".nii": itk.NiftiImageIO,
        ".nrrd": itk.NrrdImageIO,
    }
    io_class = imageio_map.get(suffix)
    
    if io_class is None:
        # Fall back to factory for unknown formats
        image_io = itk.ImageIOFactory.CreateImageIO(
            file_path_str, itk.CommonEnums.IOFileMode_ReadMode
        )
    else:
        image_io = io_class.New()

    if image_io is None:
        raise RuntimeError(f"Could not create ImageIO for {file_path}")

    image_io.SetFileName(file_path_str)
    image_io.ReadImageInformation()

    metadata = {
        "dimensions": image_io.GetNumberOfDimensions(),
        "size": [image_io.GetDimensions(i) for i in range(image_io.GetNumberOfDimensions())],
        "spacing": [image_io.GetSpacing(i) for i in range(image_io.GetNumberOfDimensions())],
        "origin": [image_io.GetOrigin(i) for i in range(image_io.GetNumberOfDimensions())],
        "pixel_type": image_io.GetPixelTypeAsString(image_io.GetPixelType()),
        "component_type": image_io.GetComponentTypeAsString(image_io.GetComponentType()),
        "number_of_components": image_io.GetNumberOfComponents(),
        "byte_order": (
            "LittleEndian"
            if image_io.GetByteOrder() == itk.CommonEnums.IOByteOrder_LittleEndian
            else "BigEndian"
        ),
    }

    return metadata