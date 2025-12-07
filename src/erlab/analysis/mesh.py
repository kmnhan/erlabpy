"""Mesh analysis and removal utilities.

This module provides functions to detect and remove periodic mesh patterns from ARPES
data acquired in fixed mode, using an experimental Fourier-based approach.
"""

import typing
from collections.abc import Iterable

import numba
import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.ndimage
import scipy.stats
import xarray as xr

import erlab


@numba.njit(cache=True)
def _get_binned_region(arr: npt.NDArray, i: int, j: int, factor: int = 1):
    x0, y0 = i * factor, j * factor
    return arr[x0 : x0 + factor, y0 : y0 + factor]


@numba.njit(cache=True)
def _bin_image(arr: npt.NDArray, factor: int = 8):
    M = arr.shape[0] // factor
    N = arr.shape[1] // factor
    resampled = np.zeros((M, N), dtype=arr.dtype)
    for i in range(M):
        for j in range(N):
            resampled[i, j] = np.mean(_get_binned_region(arr, i, j, factor))
    return resampled


def _find_local_maxima(image: npt.NDArray, min_distance: int, num_peaks: int):
    """Find peaks in a grayscale 2D image with a minimum distance between them.

    Uses a maximum filter to identify local maxima. The implementation is inspired by
    scikit-image's `peak_local_max` function but simplified for our specific use case.

    Parameters
    ----------
    image
        2D grayscale image in which to find peaks.
    min_distance
        Minimum number of pixels separating peaks.
    num_peaks
        Maximum number of peaks to return. The peaks are sorted by intensity.

    Returns
    -------
    coordinates : np.ndarray of shape (num_peaks, 2)
        Array of pixel coordinates of the detected peaks, sorted by intensity in
        descending order.
    """
    image = np.asarray(image)

    image_max = scipy.ndimage.maximum_filter(
        image,
        footprint=np.ones((2 * min_distance + 1,) * 2, dtype=bool),
        mode="nearest",
    )
    mask = image == image_max
    for i in range(2):
        mask[(slice(None),) * i + (slice(None, min_distance),)] = 0
        mask[(slice(None),) * i + (slice(-min_distance, None),)] = 0

    coordinates = np.nonzero(mask)

    # Highest peak first
    idx_maxsort = np.argsort(-image[coordinates], kind="stable")
    coordinates = np.transpose(coordinates)[idx_maxsort]

    if len(coordinates) > num_peaks:
        coordinates = coordinates[:num_peaks]

    # Ensure coords are at least separated by min_distance
    if min_distance > 0 and len(coordinates) > 1:
        accepted_coords = [coordinates[0]]
        for coord in coordinates[1:]:
            dists = (np.array(accepted_coords) - coord) ** 2
            dists = np.sqrt(dists.sum(axis=1))
            if np.all(dists >= min_distance):
                accepted_coords.append(coord)
            if len(accepted_coords) >= num_peaks:
                break
        coordinates = np.array(accepted_coords)

    return coordinates


def find_peaks(
    arr: npt.NDArray,
    *,
    bins: int = 2,
    n_peaks: int = 2,
    min_distance: int | None = None,
    plot: bool = False,
) -> npt.NDArray[np.intp]:
    """Find peaks in the FFT log magnitude image.

    Selects the upper half of the FFT magnitude image, downsamples it by the specified
    binning factor, and applies a local maximum filter to find the peaks. The detected
    peak coordinates are then mapped back to the original image resolution.

    Parameters
    ----------
    arr
        2D array-like input data that corresponds to the FFT log magnitude.
    bins
        Binning factor to pre-bin the image prior to peak finding.
    n_peaks
        Number of peaks to find (excluding the center).
    min_distance
        Minimum distance between peaks. If None, defaults to 40 // bins.
    plot
        Whether to plot the detected peaks on the input data.

    Returns
    -------
    peaks_array : np.ndarray of shape (n_peaks + 1, 2)
        Array of the coordinates of the detected peaks, including the center at index 0.
        The coordinates are in (row, column) format, and are ordered by their distance
        from the center.

    """
    # Initialize peaks array with center
    peaks_array = np.zeros((n_peaks + 1, 2), dtype=np.intp)
    peaks_array[0, 0] = arr.shape[0] // 2
    peaks_array[0, 1] = arr.shape[1] // 2

    # Extract upper half of the image and bin it
    upper_half = arr[: arr.shape[0] // 2]
    upper_half = np.r_[
        np.zeros((upper_half.shape[0] % bins, upper_half.shape[1])),
        upper_half,
    ]
    upper_half_sampled = _bin_image(upper_half, bins)
    fft_center = (upper_half_sampled.shape[1] // 2, upper_half_sampled.shape[0] - 1)

    # Locate peaks in the binned upper half, add some extra to be safe
    res = _find_local_maxima(
        upper_half_sampled,
        num_peaks=n_peaks + 5,
        min_distance=min_distance or max(1, 40 // bins),
    )
    # List for peak coordinates
    resampled_peak_idx: list[tuple[int, int]] = [tuple(reversed(x)) for x in res]

    # Sort peaks by distance to center and select top n_peaks
    resampled_peak_idx = sorted(
        resampled_peak_idx,
        key=lambda xy: np.hypot(xy[0] - fft_center[0], xy[1] - fft_center[1]),
    )[:n_peaks]

    # Map peak coordinates back to original image resolution
    for i, (x, y) in enumerate(resampled_peak_idx):
        original_region = _get_binned_region(upper_half, y, x, bins)
        if original_region.size == 0:
            continue
        py, px = np.unravel_index(np.argmax(original_region), original_region.shape)
        peaks_array[i + 1, 1] = px + x * bins
        peaks_array[i + 1, 0] = py + y * bins

        if peaks_array[i + 1, 1] <= arr.shape[0] // 2:
            peaks_array[i + 1, 1] = arr.shape[0] - 1 - peaks_array[i + 1, 1]
            peaks_array[i + 1, 0] = arr.shape[1] - 1 - peaks_array[i + 1, 0]

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(arr)
        for y, x in peaks_array:
            plt.plot(x, y, "o")
    return peaks_array


def higher_order_peaks(
    first_order: Iterable[Iterable[int]],
    order: int,
    shape: tuple[int, int],
    *,
    only_upper: bool = True,
    include_center: bool = True,
) -> npt.NDArray[np.intp]:
    """Generate peaks up to nth order from two first-order basis peaks.

    Parameters
    ----------
    first_order: array-like of shape (3, 2)
        Array of points, where first_order[0] is the center, and first_order[1] and
        first_order[2] are the two first-order peaks.
    order
        Up to which order to generate peaks.
    shape
        Shape of the image to constrain the peaks within.
    only_upper
        Whether to only generate peaks in the upper half of the image. Since the FFT
        image is Hermitian, the lower half is redundant.
    include_center
        Whether to include the center point.

    Returns
    -------
    peaks: np.ndarray of shape (M, 2), dtype=int
        Array of generated peak coordinates. M = 2 * order * (order + 1) + (3 if
        include_center else 2)
    """
    first_order = np.asarray(first_order, dtype=np.intp)
    c = first_order[0]
    b1 = first_order[1] - c
    b2 = first_order[2] - c

    pts: list[tuple[int, int]] = []
    if include_center:
        pts.append(tuple(c))

    # Add first order peaks
    pts.append(tuple(first_order[1]))
    pts.append(tuple(first_order[2]))

    # Generate higher order peaks
    order = int(order) + 1
    for nx in range(-order + 1, order):
        for ny in range(-order + 1, order):
            if max(abs(nx), abs(ny), abs(nx + ny)) > order - 1:
                continue
            if {nx, ny} == {0, 1} or {nx, ny} == {0}:
                continue
            p = c + nx * b1 + ny * b2
            if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                # Out of bounds
                continue
            if only_upper and p[0] > shape[0] // 2:
                # Skip lower half
                continue
            pts.append(tuple(p))

    # Deduplicate while preserving order
    pts = list(dict.fromkeys(pts))
    return np.asarray(pts, dtype=np.intp)


def _extract_blob_mask(S, center, roi_hw=50, k=1.0):
    H, W = S.shape
    r0, c0 = np.rint(center).astype(int)
    rs = slice(max(0, r0 - roi_hw), min(H, r0 + roi_hw + 1))
    cs = slice(max(0, c0 - roi_hw), min(W, c0 + roi_hw + 1))
    roi = S[rs, cs]

    med = np.median(roi)
    mad = scipy.stats.median_abs_deviation(roi, axis=None, scale="normal")
    th = med + k * mad

    blob = roi >= th
    if not blob.any():
        return np.zeros_like(S, bool)

    # connected component containing the ROI maximum
    rloc, cloc = np.unravel_index(np.argmax(roi), roi.shape)
    labels, nlab = scipy.ndimage.label(blob, structure=np.ones((3, 3), bool))
    if nlab == 0:
        return np.zeros_like(S, bool)
    kept = labels == labels[rloc, cloc]

    kept = scipy.ndimage.binary_fill_holes(kept)
    # kept = scipy.ndimage.binary_opening(kept, structure=np.ones((3, 3), bool))

    mask = np.zeros_like(S, bool)
    mask[rs, cs] = kept
    return mask


def _fit_moments(mask, center):
    """Second-moment ellipse fit from a binary mask (fftshifted coords)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return (0, 0), 2.0, 2.0, 0.0  # fallback
    y0, x0 = center
    x, y = (xs - x0, ys - y0)
    Cxx, Cxy, Cyy = (x * x).mean(), (x * y).mean(), (y * y).mean()

    # Eigenvalue decomposition
    trace = Cxx + Cyy
    det = Cxx * Cyy - Cxy * Cxy
    tmp = np.sqrt(max(trace * trace / 4 - det, 0.0))

    # Principal axis angle
    theta = 0.5 * np.arctan2(2 * Cxy, (Cxx - Cyy + 1e-12))

    # Convert variances to sigmas (soften)
    sig_major = np.sqrt(max(trace / 2 + tmp, 1e-3)) * 1.25
    sig_minor = np.sqrt(max(trace / 2 - tmp, 1e-3)) * 1.25

    return (y0, x0), sig_major, sig_minor, float(theta)


def _rotated_gaussian_notch(
    shape, center, sig_major, sig_minor, theta, *, strength=1.0
):
    H, W = shape
    y, x = np.indices((H, W))
    yc, xc = center
    dx, dy = (x - xc), (y - yc)
    ct, st = np.cos(theta), np.sin(theta)
    u = ct * dx + st * dy  # along major axis
    v = -st * dx + ct * dy  # along minor axis
    G = np.exp(-0.5 * ((u / sig_major) ** 2 + (v / sig_minor) ** 2))
    return 1.0 - strength * G


@numba.njit(cache=True)
def _symmetrize_hermitian(H):
    m, n = H.shape
    out = np.empty_like(H)
    for i in range(m):
        ii = m - 1 - i
        for j in range(n):
            jj = n - 1 - j
            out[i, j] = min(H[i, j], H[ii, jj])
    return out


def auto_correct_curvature(
    darr: xr.DataArray, poly_deg: int = 4
) -> tuple[xr.DataArray, xr.DataArray]:
    """Automatically correct curvature in the energy axis of ARPES data.

    Some analyzers like the Scienta DA30L optionally outputs data with software
    corrected Fermi edge compensating for the straight analyzer slit. This also warps
    the mesh pattern and blurs the peaks in the FFT image, making mesh removal
    difficult. This function estimates the curvature by finding the energy positions
    where the derivative of the intensity profile changes sign along the angular axis,
    and fits a polynomial to this profile. The data is then shifted back to undo the
    effect. Then, the data is trimmed to remove edge artifacts, and padded back with
    edge values to the original size.

    This has only been tested on data produced by the DA30L analyzer.

    Parameters
    ----------
    darr
        Input DataArray. Must be 2D with 'alpha' and 'eV' dimensions.
    poly_deg
        Degree of the polynomial to fit to the curvature profile. By default 4.

    Returns
    -------
    shift_arr : xr.DataArray
        Array of shifts applied to each angular position to correct the curvature.
    corrected : xr.DataArray
        Curvature-corrected DataArray.

    """
    darr_partial = darr.isel(alpha=slice(10, -10))
    # Get sign change of derivative w.r.t. eV
    edge = darr_partial.diff("eV") > 0

    # Data right at this index is nonzero, one step lower is zero
    correction_profile_idx = typing.cast("xr.DataArray", edge.argmax("eV"))

    # Fit polynomial to the profile
    step = float(np.abs(edge.eV[1] - edge.eV[0]))
    profile_fit = correction_profile_idx.polyfit("alpha", deg=poly_deg)
    poly_values = xr.polyval(darr.alpha, profile_fit).polyfit_coefficients
    shift_idx_arr = poly_values.round()

    max_idx = int(shift_idx_arr.argmax("alpha"))
    cutoff_index = int(shift_idx_arr.max()) + 2

    # Get shift in eV
    shift_arr = (shift_idx_arr.values[max_idx] - poly_values) * step
    shifted = erlab.analysis.transform.shift(
        darr, shift_arr, "eV", shift_coords=False, order=3, cval=0.0, prefilter=True
    ).clip(min=0)

    shifted_trimmed = shifted.copy()
    original_dim_order = list(darr.dims)
    shifted_trimmed = shifted_trimmed.isel(eV=slice(cutoff_index, -5)).transpose(
        "alpha", "eV"
    )

    shifted = (
        shifted.transpose("alpha", "eV")
        .copy(
            data=np.pad(
                shifted_trimmed.values,
                ((0, 0), (cutoff_index, 5)),
                mode="edge",
            )
        )
        .transpose(*original_dim_order)
    )

    return shift_arr, shifted


def pad_and_taper(arr: npt.NDArray, n: int) -> npt.NDArray:
    """Pad the input array and apply edge tapering using a Tukey window.

    Parameters
    ----------
    arr : np.ndarray
        2D input array to be padded and tapered.
    n : int
        Number of pixels to pad around the array.

    Returns
    -------
    padded_tapered : np.ndarray
        Padded and tapered array.
    """
    if n == 0:
        return arr
    padded = np.pad(arr, n, mode="edge")

    Hp, Wp = padded.shape
    alpha_y = min(2.0 * n / max(Hp - 1, 1), 1.0)
    alpha_x = min(2.0 * n / max(Wp - 1, 1), 1.0)

    wy = scipy.signal.windows.tukey(Hp, alpha=alpha_y, sym=True)
    wx = scipy.signal.windows.tukey(Wp, alpha=alpha_x, sym=True)
    win2d = wy[:, np.newaxis] * wx[np.newaxis, :]
    return padded * win2d


def unpad(arr: npt.NDArray, n: int) -> npt.NDArray:
    """Remove padding from the input array.

    Parameters
    ----------
    arr : np.ndarray
        2D input array to be unpadded.
    n : int
        Number of pixels to remove from each side.

    Returns
    -------
    unpadded : np.ndarray
        Unpadded array.
    """
    if n == 0:
        return arr
    return arr[n:-n, n:-n]


def remove_mesh(
    darr: xr.DataArray,
    *,
    first_order_peaks: Iterable[Iterable[int]] | None = None,
    order: int = 3,
    n_pad: int = 90,
    roi_hw: int = 25,
    k: float = 0.5,
    feather: float = 3.0,
    undo_edge_correction: bool = False,
    full_output: bool = False,
    method: typing.Literal["constant", "gaussian", "circular"] = "constant",
) -> (
    tuple[xr.DataArray, xr.DataArray]
    | tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray | None,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
    ]
):
    """Remove mesh patterns from ARPES data using notch filtering in the FFT domain.

    This function identifies mesh patterns in the FFT log-magnitude of the input data,
    creates notch filters to suppress these patterns, and applies the filters to remove
    the mesh from the data.

    This method is experimental and may not work perfectly for all cases.

    Parameters
    ----------
    darr
        Input DataArray with 'alpha' and 'eV' dimensions. All dimensions other than
        these will be averaged over before extracting the mesh. The corrected data will
        retain all original dimensions.
    first_order_peaks
        Coordinates of the two first-order mesh peaks in (row, column) format. If
        `None`, auto-detection will be performed. There should be three rows, where the
        first row is the center index of the FFT image, and the next two rows are the
        first-order peaks.
    order
        Up to which order of mesh peaks to remove.
    n_pad
        Number of pixels to pad around the image before FFT to reduce edge artifacts.
        Edge tapering is also applied.
    roi_hw
        Half-width of the region of interest around each peak for mask creation.
    k
        Thresholding parameter for blob extraction around each peak. Higher values
        result in smaller blobs.
    feather
        Amount of feathering to apply to the notch masks. Higher values result in
        smoother masks, which can help reduce ringing artifacts, but may also leave some
        mesh residuals.
    undo_edge_correction
        Whether to automatically correct curvature in the energy axis before mesh
        removal. This is useful for data from analyzers that apply software edge
        correction, which can warp the mesh pattern. Currently only tested with Scienta
        DA30L data.
    full_output
        Whether to return additional diagnostic outputs. See Returns section.
    method
        Method for creating the notch masks. Options are:

        - "constant": Simple binary notch (hard cut with feathering).

        - "gaussian": Gaussian-shaped notch fitted to the blob shape.

        - "circular": Circular notch with radius based on blob size.

        It is recommended to use "constant" for most cases. Needs more testing.

    Returns
    -------
    corrected : xr.DataArray
        Mesh-corrected DataArray with the same shape as the input.
    mesh : xr.DataArray
        Extracted mesh DataArray with the same shape as the input.
    shift_arr : xr.DataArray
        Array of shifts applied to correct curvature. (Returned when full_output is
        `True`.)
    log_magnitude : np.ndarray
        Log-magnitude of the FFT before mesh removal. (Returned when full_output is
        `True`.)
    log_magnitude_corr : np.ndarray
        Log-magnitude of the FFT after mesh removal. (Returned when full_output is
        `True`.)
    peaks : np.ndarray
        Coordinates of all detected mesh peaks used for notch creation. Note that since
        the FFT image is Hermitian, to get the full set of peaks, one needs to mirror
        these coordinates about the center. (Returned when full_output is `True`.)
    mask : np.ndarray
        Final combined notch mask applied in the FFT domain. (Returned when full_output
        is `True`.)

    """
    core_dims = ("alpha", "eV")
    if not all(dim in darr.dims for dim in core_dims):
        raise ValueError("Input DataArray must have 'alpha' and 'eV' dimensions.")

    other_dims = tuple(dim for dim in darr.dims if dim not in core_dims)

    # DataArray to extract the mesh from
    original = darr.mean(other_dims).transpose(*core_dims).compute()

    shift_arr: xr.DataArray | None = None

    if undo_edge_correction:
        shift_arr, original = auto_correct_curvature(original)

    image: npt.NDArray = original.fillna(0).values

    image = pad_and_taper(image, n_pad)

    # Compute FFT and log-magnitude
    fft = scipy.fft.fftshift(scipy.fft.fft2(image))
    log_magnitude = np.log(np.abs(fft).clip(min=1e-15))

    # We will apply notch on log of FFT to treat the multiplicative mesh as additive
    # This allows us to correctly remove mesh from data with high contrast (sharp bands)
    log_image = np.log(image.clip(min=1e-15))
    log_fft = scipy.fft.fftshift(scipy.fft.fft2(log_image))

    # If peaks are not provided, find them
    if first_order_peaks is None:
        first_order_peaks = find_peaks(log_magnitude, n_peaks=2, plot=False) - n_pad
    first_order_peaks = np.asarray(first_order_peaks, dtype=np.intp) + n_pad

    # Get all peaks up to specified order
    peaks = higher_order_peaks(
        first_order_peaks,
        order=order,
        shape=log_magnitude.shape,
        only_upper=True,
        include_center=False,
    )

    # Initialize mask
    mask = np.ones(log_magnitude.shape, np.float64)
    for peak in peaks:
        peak_blob = _extract_blob_mask(log_magnitude, peak, roi_hw=roi_hw, k=k)
        match method:
            case "constant":
                mask = mask * (1 - peak_blob)
            case "gaussian":
                gaussian_moments = _fit_moments(peak_blob, peak)
                mask = mask * (
                    _rotated_gaussian_notch(log_magnitude.shape, *gaussian_moments)
                )
            case "circular":
                blob_radius = np.sqrt(peak_blob.sum() / np.pi)
                y0, x0 = peak
                H, W = log_magnitude.shape
                y, x = np.indices((H, W))
                r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                circular_notch = r >= blob_radius
                mask = mask * circular_notch.astype(np.float64)
            case _:
                raise ValueError(f"Unknown mesh removal method: {method}")

    mask = _symmetrize_hermitian(mask)

    match method:
        case "constant":
            mask = 1 - scipy.ndimage.binary_opening(
                1 - mask, structure=np.ones((3, 3), bool)
            )

    # Feather the mask
    mask = scipy.ndimage.gaussian_filter(mask.astype(np.float64), sigma=feather)

    mesh_arr = 1 / np.exp(
        scipy.fft.ifft2(scipy.fft.ifftshift(log_fft * (1 - mask))).real
    ).clip(min=1e-15)

    mesh = original.copy(data=unpad(mesh_arr, n_pad))

    if undo_edge_correction:
        mesh = erlab.analysis.transform.shift(
            mesh,
            -typing.cast("xr.DataArray", shift_arr),
            "eV",
            shift_coords=False,
            order=3,
            cval=0.0,
            prefilter=True,
        ).clip(min=0)

    mesh = mesh.astype(darr.dtype)
    corrected = darr * mesh
    if full_output:
        log_magnitude_corr = np.log(
            np.abs(scipy.fft.fftshift(scipy.fft.fft2(image * mesh_arr))).clip(min=1e-15)
        )
        return (
            corrected,
            mesh,
            shift_arr,
            unpad(log_magnitude, n_pad),
            unpad(log_magnitude_corr, n_pad),
            peaks - n_pad,
            unpad(mask, n_pad),
        )

    return corrected, mesh
