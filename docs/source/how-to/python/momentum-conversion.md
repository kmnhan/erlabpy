# Momentum conversion

Use these guides to change the assigned configuration, convert angle-resolved data
to momentum space, compare measurements on a common momentum grid, add momentum
coordinates to a measured cut, and convert $h\nu$–dependent scans.

(how-to-python-change-configuration)=

## Changing the assigned configuration

Use this procedure when one endstation can acquire data in more than one of the four
physical configurations in {ref}`nomenclature`, and the file does not uniquely identify
the configuration used for one measurement. Do not select a configuration from the
appearance of the measured intensity.

Two common cases are an analyzer with a slit that rotates about the lens axis and a
deflector-equipped analyzer that can acquire a map either with the deflector or by
moving a physical sample angle. For example, the analyzer at ALS BL7 can change its
slit orientation. A horizontal-slit tilt map from that endstation uses configuration 2,
whereas the MAESTRO loader uses configuration 3, a vertical-slit deflector map, as its
default.

Do not use {meth}`xarray.DataArray.kspace.as_configuration` to repair arbitrary names
from an incorrect loader implementation. Data from a fixed-geometry endstation must
already have the correct configuration and coordinate names after loading. Correct the
loader when it does not.

A loader normally assigns the configuration. When you construct an angle-resolved
{class}`DataArray <xarray.DataArray>` directly and it has no `configuration` attribute,
assign the known configuration once:

```python
import erlab

data.kspace.configuration = erlab.constants.AxesConfiguration.Type1DA
```

Use {meth}`xarray.DataArray.kspace.as_configuration` only when the data already has a
configuration and the measurement used another supported physical configuration.

Inspect the configuration assigned by the loader:

```python
data.kspace.configuration
```

For the ALS BL7 case above, create a {class}`DataArray <xarray.DataArray>` for the
horizontal-slit configuration:

```python
configured = data.kspace.as_configuration(
    erlab.constants.AxesConfiguration.Type2,
)
```

The method translates the standard coordinate names by their physical roles. In a
change from configuration 3 to configuration 2, `chi` becomes `xi`, `xi` becomes
`beta`, and `beta` becomes `beta_deflector`. Only coordinates present in the data are
renamed.

The method also changes the `configuration` attribute. It does not rotate, interpolate,
or otherwise transform the intensity. It returns a copy, and the original
{class}`DataArray <xarray.DataArray>` is unchanged. Use `configured` for the remaining
analysis.

This translation assumes the standard vertical-cryostat geometries in
{ref}`nomenclature`. For a different geometry, rename the physical angle coordinates
and set the `configuration` attribute explicitly. See
{attr}`xarray.DataArray.kspace.configuration` and
{meth}`xarray.DataArray.kspace.as_configuration` for the complete interface.

(how-to-python-convert-angle-data)=

## Converting to momentum space

Use this guide after the data follows {ref}`data-conventions` and after you set the
experimental configuration and normal emission angles.

Store the measured normal emission position and work function, then convert:

```python
conversion_input = data.copy()
conversion_input.kspace.set_normal(
    alpha=alpha_normal,
    beta=beta_normal,
    delta=azimuthal_offset,
)
conversion_input.kspace.work_function = work_function

converted = conversion_input.kspace.convert()
```

`alpha_normal` and `beta_normal` are the data coordinates that correspond to normal
emission. Use {ref}`how-to-python-change-configuration` first when a variable-geometry
endstation acquired the measurement in a different configuration from the loader
default.

To control the output grid, supply momentum bounds and a target step through the
`resolution` argument:

```python
converted = conversion_input.kspace.convert(
    bounds={"kx": (-0.5, 0.5), "ky": (-0.5, 0.5)},
    resolution={"kx": 0.01, "ky": 0.01},
)
```

The final step can differ slightly from the target because the grid contains an integer
number of intervals between both bounds. Pass explicit momentum coordinate arrays when
the target values must be exact.

```{eval-rst}
.. plot:: how_to/momentum_conversion.py convert_angle_resolved_data
   :include-source: false
   :alt: Constant energy map in angle coordinates, on the automatic momentum grid, and on a specified momentum grid
```

Use {doc}`../../tutorials/python/index` for the basic
generated-data conversion. See {doc}`momentum conversion
<../../explanation/momentum-conversion>` before selecting geometry, offsets, or output
sampling. See {meth}`xarray.DataArray.kspace.convert` for all conversion arguments.
Use {ref}`guide-ktool` to adjust these parameters interactively and copy the resulting
configuration, offsets, and conversion call to Python.

(how-to-python-convert-common-grid)=

## Converting measurements to a common momentum grid

Use the same target coordinates when you must compare or combine converted
measurements point by point. First, prepare each measurement with its experimental
configuration and energy parameters:

```python
import numpy as np

conversion_input = data.copy()
conversion_input.kspace.set_normal(
    alpha=alpha_normal,
    beta=beta_normal,
    delta=azimuthal_offset,
)
conversion_input.kspace.work_function = work_function
```

Define the target grid once. Pass the coordinate arrays to each conversion:

```python
target_kx = np.linspace(-0.5, 0.5, 101)
target_ky = np.linspace(-0.5, 0.5, 101)

converted = conversion_input.kspace.convert(
    kx=target_kx,
    ky=target_ky,
)
```

An explicit coordinate array replaces the automatic bounds and spacing for that axis.
Points outside the measured coverage contain missing values. Select a common valid
region before you compare intensity between measurements.

See {meth}`xarray.DataArray.kspace.convert` for the accepted target coordinates. See
{doc}`../../explanation/momentum-conversion` for the distinction between interpolation
spacing and experimental momentum resolution.

(how-to-python-convert-coordinates-only)=

## Converting coordinates only

Select the cut in angle space. Then calculate momentum coordinates without
interpolating its intensity:

```python
cut = data.qsel(beta=-10)
cut_with_momentum = cut.kspace.convert_coords()
```

Convert `data` to momentum space and select the same energy from the cut and converted
data:

```python
converted_map = data.kspace.convert()
cut_path = cut_with_momentum.qsel(eV=-0.3)
constant_energy_map = converted_map.qsel(eV=-0.3)
```

The figure below shows the calculated cut trajectory on the converted constant energy
surface. See {doc}`../plotting/cut-trajectories` for the Python plotting code and Figure
Composer steps.

```{eval-rst}
.. plot:: how_to/momentum_conversion.py overlay_cut_path
   :include-source: false
   :alt: Angular cut trajectory overlaid on a converted constant energy surface
```

The cut keeps its measured dimensions and intensity values. The added `kx` and `ky`
coordinates describe its path through the converted constant energy surface.

Use the same experimental configuration, work function, and normal emission position
for `cut_with_momentum` and `converted_map`. Otherwise, the cut trajectory and surface
use different conversion parameters. See
{meth}`xarray.DataArray.kspace.convert_coords` for the returned coordinates. See
{doc}`momentum conversion
<../../explanation/momentum-conversion>` for the difference between adding momentum
coordinates with {meth}`convert_coords <xarray.DataArray.kspace.convert_coords>` and
interpolating intensity with {meth}`convert <xarray.DataArray.kspace.convert>`.

(how-to-python-convert-photon-energy-scan)=

## Converting hν–dependent scans

Use this guide when `data` contains an $h\nu$–dependent scan with the coordinates and
attributes listed in {ref}`data-conventions`. Set the experimental configuration,
normal emission angles, work function, and inner potential before conversion.

Set the inner potential, then convert the data:

```python
conversion_input = data.copy()
conversion_input.kspace.inner_potential = inner_potential
converted = conversion_input.kspace.convert()
```

Use an `inner_potential` that is consistent with the measured $k_z$ periodicity and the
known reciprocal lattice. Do not copy an example value into an experimental analysis.

To calculate $k_z$ positions for selected photon energies without another interpolation,
use the converted data:

```python
kz_values = converted.kspace.hv_to_kz([30, 45, 60]).qsel(eV=-0.3)
```

```{eval-rst}
.. plot:: how_to/momentum_conversion.py convert_hv_dependent_scan
   :include-source: false
   :alt: hν-dependent scan in angle and momentum coordinates
```

Check the converted coordinate ranges against the expected reciprocal-lattice period.
See {meth}`xarray.DataArray.kspace.convert` and
{meth}`xarray.DataArray.kspace.hv_to_kz` for accepted arguments. See
{doc}`../../explanation/momentum-conversion` for the conventions for geometry, energy,
and inner potential.

(how-to-python-mark-photon-energies)=

## Annotating photon energies

Use the converted $h\nu$–dependent scan to calculate the $k_z$ values for selected
photon energies. Select the binding energy that you will show:

```python
photon_energies = [30, 45, 60]
binding_energy = -0.3
kz_values = converted.kspace.hv_to_kz(photon_energies).qsel(
    eV=binding_energy,
)
```

The figure below shows these coordinates as constant-photon-energy curves. See
{doc}`../plotting/photon-energy-annotations` for the Python plotting code and Figure
Composer steps.

```{eval-rst}
.. plot:: how_to/momentum_conversion.py annotate_photon_energies
   :include-source: false
   :alt: Converted constant energy surface with calculated kz values for three photon energies
```

The lines use the stored geometry, work function, and inner potential. They are
calculated coordinates. They do not show measured intensity at a new photon energy.

See {meth}`xarray.DataArray.kspace.hv_to_kz` for accepted photon energies. See
{doc}`../../explanation/momentum-conversion` for the role of the inner potential in the
calculated $k_z$ coordinate.
