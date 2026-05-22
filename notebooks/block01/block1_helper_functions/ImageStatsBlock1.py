# Set of functions to compare some stats and characteristics of two MR images.

# This work contains several main functions with associated helper functions:
# 1. IntensityHistogram: Plots the histogram of voxel intensities for each image.

# 2. NonZeroVoxelCount: Counts the number of non-zero voxels in each image.

# 3. FFTSpectra: compute the 3D Fast Fourier Transform (FFT) of each image and plot the magnitude spectra.

# 4. IntensityRange: Compares the minimum and maximum voxel intensities between the two images.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


# Taken from ImageViewerBlock1.py, with some modifications to allow for two images to be compared in the same histogram plot.
def load_mri_volume(image_path: Path | str) -> np.ndarray:
    """Load a NIfTI image and return a 3D array ready for visualization."""

    image = nib.load(str(image_path))
    data = np.asarray(image.get_fdata())

    if data.ndim == 4:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D or 4D MRI volume, but found an array with shape {data.shape}."
        )

    data = np.nan_to_num(data, copy=False)
    # Rotate 180 degrees to match the orientation of the sample image]
    return data
# Function 1: IntensityHistogram
def IntensityHistogram(img1, img2):
    """
    Plots the histogram of voxel intensities for each image.
    
    Parameters:
    img1 (numpy array): First MRI image data.
    img2 (numpy array): Second MRI image data.
    
    Returns:
    None: Displays the histogram plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Images if images are paths, strings or others, load them using the helper function

    if isinstance(img1, str):
        img1 = load_mri_volume(img1)

    if isinstance(img2, str):
        img2 = load_mri_volume(img2)
    
    img1 = load_mri_volume(img1)
    img2 = load_mri_volume(img2)


    # Flatten the images to 1D arrays
    img1_flat = img1.flatten()
    # Normalise both images (i.e demean and divide by standard deviation) to make them more comparable  
    img1_flat = (img1_flat - np.mean(img1_flat)) / np.std(img1_flat)
    img2_flat = img2.flatten()
    img2_flat = (img2_flat - np.mean(img2_flat)) / np.std(img2_flat)

    # Take only real values (in case of complex numbers from FFT)
    img1_flat = np.real(img1_flat)
    img2_flat = np.real(img2_flat)
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(img1_flat, bins=50, color='blue', alpha=0.7)
    plt.title('Intensity Histogram - Image 1')
    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(img2_flat, bins=50, color='orange', alpha=0.7)
    plt.title('Intensity Histogram - Image 2')
    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Function 2: NonZeroVoxelCount
def NonZeroVoxelCount(img1, img2):
    """
    Counts the number of non-zero voxels in each image.
    
    Parameters:
    img1 (numpy array): First MRI image data.
    img2 (numpy array): Second MRI image data.
    
    Returns:
    tuple: A tuple containing the count of non-zero voxels for each image (count_img1, count_img2).
    """
    import numpy as np

    # Images if images are paths, strings or others, load them using the helper function

    if isinstance(img1, str):
        img1 = load_mri_volume(img1)

    if isinstance(img2, str):
        img2 = load_mri_volume(img2)
    
    img1 = load_mri_volume(img1)
    img2 = load_mri_volume(img2)

    # Flatten for speed
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    count_img1 = np.count_nonzero(img1_flat)
    count_img2 = np.count_nonzero(img2_flat)

    # Divide by total number of voxels in each image to get a percentage of non-zero voxels, which is more comparable across different image sizes
    count_img1 = count_img1 / img1_flat.size
    count_img2 = count_img2 / img2_flat.size

    # Print the results
    print(f"Image 1 - Non-zero voxel count: {count_img1*100:.2f}%")
    print(f"Image 2 - Non-zero voxel count: {count_img2*100:.2f}%")

    return count_img1, count_img2

# Dev function, not used in practical, but could be useful for more detailed analysis of the frequency content of the images.
def FFTSpectra(img1, img2, axis=0):
    """
    Plot detailed FFT views for two 3D MRI volumes:
    - a central 2D FFT power slice
    - a 1D center-line profile through that slice
    """
    import numpy as np
    import matplotlib.pyplot as plt

    img1 = load_mri_volume(img1)
    img2 = load_mri_volume(img2)

    def fft_views(vol, axis=0):
        vol = np.asarray(vol, dtype=np.float64)
        vol = vol - np.mean(vol)

        fft_vol = np.fft.fftn(vol)
        fft_vol = np.fft.fftshift(fft_vol)
        power = np.abs(fft_vol) ** 2
        power = np.log1p(power)

        if axis == 0:
            sl = power[power.shape[0] // 2, :, :]
            line = sl[sl.shape[0] // 2, :]
        elif axis == 1:
            sl = power[:, power.shape[1] // 2, :]
            line = sl[sl.shape[0] // 2, :]
        else:
            sl = power[:, :, power.shape[2] // 2]
            line = sl[sl.shape[0] // 2, :]

        return sl, line

    sl1, line1 = fft_views(img1, axis=axis)
    sl2, line2 = fft_views(img2, axis=axis)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(sl1, cmap="gray", origin="lower")
    plt.title("FFT Power Slice - Image 1")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(sl2, cmap="gray", origin="lower")
    plt.title("FFT Power Slice - Image 2")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.plot(line1, color="blue")
    plt.title("Center Line Profile - Image 1")
    plt.xlabel("Frequency index")
    plt.ylabel("log(1 + power)")

    plt.subplot(2, 2, 4)
    plt.plot(line2, color="orange")
    plt.title("Center Line Profile - Image 2")
    plt.xlabel("Frequency index")
    plt.ylabel("log(1 + power)")

    plt.tight_layout()
    plt.show()


# Function 4: IntensityRange
def IntensityRange(img1, img2):
    """
    Compares the minimum and maximum voxel intensities between the two images.
    
    Parameters:
    img1 (numpy array): First MRI image data.
    img2 (numpy array): Second MRI image data.
    
    Returns:
    tuple: A tuple containing the intensity range for each image ((min_img1, max_img1), (min_img2, max_img2)).
    """
    import numpy as np

    # Images if images are paths, strings or others, load them using the helper function

    if isinstance(img1, str):
        img1 = load_mri_volume(img1)

    if isinstance(img2, str):
        img2 = load_mri_volume(img2)
    
    img1 = load_mri_volume(img1)
    img2 = load_mri_volume(img2)

    # Only take the abs


    min_img1 = np.min(img1)
    max_img1 = np.max(img1)
    min_img2 = np.min(img2)
    max_img2 = np.max(img2)

    print(f"Image 1 - Intensity Range: [{min_img1}, {max_img1}]")
    print(f"Image 2 - Intensity Range: [{min_img2}, {max_img2}]")

    return (min_img1, max_img1), (min_img2, max_img2)

# Function 5: CNR and SNR (WIP, no ROI masks provided in sample data, will run in FSLEYES and add some ROI masks to the sample data for testing)
def CNR_SNR(img1, img2, roi_mask):
    """
    Computes the Contrast-to-Noise Ratio (CNR) and Signal-to-Noise Ratio (SNR) for two images given a region of interest (ROI) mask.
    
    Parameters:
    img1 (numpy array): First MRI image data.
    img2 (numpy array): Second MRI image data.
    roi_mask (numpy array): A binary mask defining the region of interest for CNR/SNR calculation.
    
    Returns:
    tuple: A tuple containing the CNR and SNR values for each image ((CNR_img1, SNR_img1), (CNR_img2, SNR_img2)).
    """
    import numpy as np

    # Images if images are paths, strings or others, load them using the helper function

    if isinstance(img1, str):
        img1 = load_mri_volume(img1)

    if isinstance(img2, str):
        img2 = load_mri_volume(img2)
    
    img1 = load_mri_volume(img1)
    img2 = load_mri_volume(img2)

    signal_img1 = np.mean(img1[roi_mask > 0])
    noise_img1 = np.std(img1[roi_mask == 0])
    signal_img2 = np.mean(img2[roi_mask > 0])
    noise_img2 = np.std(img2[roi_mask == 0])

    snr_img1 = signal_img1 / noise_img1 if noise_img1 > 0 else np.inf
    snr_img2 = signal_img2 / noise_img2 if noise_img2 > 0 else np.inf
    cnr = abs(signal_img1 - signal_img2) / np.sqrt(noise_img1**2 + noise_img2**2) if (noise_img1 > 0 and noise_img2 > 0) else np.inf

    print(f"Image 1 - SNR: {snr_img1:.2f}")
    print(f"Image 2 - SNR: {snr_img2:.2f}")
    print(f"CNR between Image 1 and Image 2: {cnr:.2f}")

    return (cnr, snr_img1), (cnr, snr_img2)


# Functin 6: Compare IDPs:
def compareIDPs(path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import ast

    # Load CSV
    idp_df = pd.read_csv(path)

    # Parse T1_FIRST_ columns
    for col in idp_df.columns:
        if col.startswith("T1_FIRST_"):

            def parse_value(x):
                if pd.isna(x):
                    return np.nan

                if isinstance(x, str):
                    x = x.strip()

                    # Handle "[1,2,3]" format
                    if x.startswith("[") and x.endswith("]"):
                        vals = ast.literal_eval(x)
                    else:
                        vals = [v.strip() for v in x.split(",")]

                    arr = np.array(vals, dtype=float)

                    # Reduce vector to scalar for plotting
                    return arr.mean()

                return float(x)

            idp_df[col] = idp_df[col].apply(parse_value)

    # Extract scanner/image ID
    idp_df["image_id"] = (
        idp_df["scanner_name"]
        .astype(str)
        .apply(lambda x: x.split("_")[0])
    )

    # Reshape dataframe:
    # x-axis = IDP name
    # colour/group = scanner
    idp_cols = [col for col in idp_df.columns if col.startswith("T1_FIRST_")]

    plot_df = idp_df.melt(
        id_vars="image_id",
        value_vars=idp_cols,
        var_name="IDP",
        value_name="Value"
    )

    # Pivot so scanners become columns
    pivot_df = plot_df.pivot(
        index="IDP",
        columns="image_id",
        values="Value"
    )

    # Plot
    ax = pivot_df.plot(
        kind="bar",
        figsize=(14, 6)
    )

    plt.title("Comparison of IDP Values by Scanner")
    plt.xlabel("IDP")
    plt.ylabel("Value")
    plt.legend(title="Scanner")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(pivot_df)