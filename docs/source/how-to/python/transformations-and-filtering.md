# Transformations and filtering

Use these guides to rotate maps and volumes, shift spectra to correct drift, apply
known symmetries, add Gaussian broadening to simulations, and visualize dispersive
features.

(how-to-python-rotate-map)=

## Rotate maps and volumes

Determine the rotation angle and center from the experimental geometry or a visible
reference feature. Rotate the two evenly sampled momentum dimensions while preserving
the remaining dimensions:

```python
import erlab.analysis as era

rotated = era.transform.rotate(
    data,
    angle=25.0,
    axes=("ky", "kx"),
    center={"ky": 0.0, "kx": 0.0},
    reshape=True,
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py rotate_map
   :include-source: false
   :alt: Constant energy maps at three energies before and after a 25 degree rotation
```

Set `reshape=False` when the output must keep the original extent. Determine the
rotation center and angle before using the result. Rotation requires interpolation
when the new grid does not coincide with the measured points. Repeated rotations can
compound this effect. See {func}`erlab.analysis.transform.rotate` for interpolation,
reshape, and fill-value arguments.

(how-to-python-align-spectra-with-offsets)=

## Shift spectra to correct drift

Use {func}`erlab.analysis.transform.shift` when `energy_offsets` contains a verified
energy offset for every value of another coordinate. The offset
{class}`DataArray <xarray.DataArray>` broadcasts
across the remaining dimensions:

```python
import erlab.analysis as era

aligned = era.transform.shift(
    spectra,
    shift=-energy_offsets,
    along="eV",
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py align_spectra_with_offsets
   :include-source: false
   :alt: First and last spectra and the fitted energy offsets before and after alignment
```

The fitted edge must follow the displacement before alignment. After the shift, the
edge must coincide with 0 eV in both endpoint spectra and across the $h\nu$ series.

Ensure that `energy_offsets` uses the same energy units and compatible coordinates as
`spectra`. Derive the offsets from an appropriate reference edge or another established
energy reference. Do not infer the energy correction from dispersing bands in the
sample data.

If the shifts would remove valid data at the original coordinate bounds, expand the
output coordinate range:

```python
aligned = era.transform.shift(
    spectra,
    shift=-energy_offsets,
    along="eV",
    shift_coords=True,
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py preserve_shifted_coordinate_range
   :include-source: false
   :alt: Aligned hν series on the original and expanded energy coordinate ranges
```

The fitted offsets record a displacement. They do not establish its physical cause.
See {func}`erlab.analysis.transform.shift` for coordinate and interpolation arguments.

(how-to-python-apply-symmetry)=

## Reflection symmetrization

Use reflection symmetrization when the data must be combined with its reflection about
a known coordinate:

```python
import erlab.analysis as era

symmetrized = era.transform.symmetrize(
    data,
    dim="kx",
    center=0.0,
    average=True,
)
antisymmetrized = era.transform.symmetrize(
    data,
    dim="kx",
    center=0.0,
    subtract=True,
    average=True,
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py apply_symmetry
   :include-source: false
   :alt: Input, reflection-symmetrized, and antisymmetric ARPES cuts
```

Use a symmetry center established independently from the transformation. Do not use
symmetrized intensity to determine normal emission or to justify a reflection symmetry
that is not known for the measured system.

See {func}`erlab.analysis.transform.symmetrize` for coordinate and output-range
arguments.

(how-to-python-apply-rotational-symmetry)=

## Rotational n-fold symmetrization

Use rotational symmetrization when a map or volume must be averaged over equivalent
in-plane rotations. Supply the symmetry order, momentum dimensions, and rotation center:

```python
import erlab.analysis as era

symmetrized = era.transform.symmetrize_nfold(
    data,
    6,
    axes=("kx", "ky"),
    center={"kx": 0.0, "ky": 0.0},
    reshape=True,
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py apply_rotational_symmetry
   :include-source: false
   :alt: Partial constant energy surfaces at three energies beside the surfaces averaged over six rotations about the zone center
```

Set `reshape=False` when the output must keep the original grid. Determine the rotation
center and symmetry order independently. Compare the result with the measured map. Do
not use the averaged result to establish the symmetry that you supplied.

See {func}`erlab.analysis.transform.symmetrize_nfold` for coordinate, interpolation,
and output-range arguments.

(how-to-python-broaden-simulated-data)=

## Gaussian convolution

Supply the Gaussian standard deviation in the units of each coordinate:

```python
import erlab.analysis as era

broadened = era.image.gaussian_filter(
    simulated_data,
    sigma={"eV": 0.01, "alpha": 0.2},
)
```

```{eval-rst}
.. plot:: how_to/transformations_and_filtering.py gaussian_convolution
   :include-source: false
   :alt: Simulated ARPES intensity before and after Gaussian convolution
```

See {func}`erlab.analysis.image.gaussian_filter` for dimension selection and boundary
handling.

(how-to-python-enhance-dispersive-features)=

## Visualizing dispersive features

Open the two-dimensional cut in `dtool`:

```python
import erlab.interactive as eri

eri.dtool(data)
```

Use the interpolation and smoothing controls before you select a derivative-based
method. Compare the processed result with the source intensity in the same window.
After you select the parameters, copy the generated calculation code or open the
result in ImageTool from the plot context menu.

See {func}`erlab.interactive.dtool` for the Python entry point. See the
{ref}`dtool reference <guide-dtool>` for its methods, controls, and ImageTool
integration.
