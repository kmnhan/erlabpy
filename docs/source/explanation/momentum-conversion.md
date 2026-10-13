# Momentum conversion

ERLabPy can calculate momentum coordinates without changing the measured sampling. It
can also interpolate the intensity onto a regular momentum grid. Both operations use
the same momentum conversion functions. The mapping functions use the complete
trigonometric geometry without a small-angle approximation.

## Conversion inputs

{ref}`data-conventions` defines the expected names, units, and storage locations.

| Input | Role in the mapping functions |
| --- | --- |
| `configuration` | Selects the mapping functions and assigns the analyzer-slit direction to `kx` or `ky` |
| `alpha` and `beta` | Supply the measured angular coordinates |
| `xi`, `chi`, and angular offsets | Set the sample orientation and normal emission position; the `delta` offset rotates the in-plane momentum axes |
| `eV`, `hv`, and `analyzer_work_function` | Determine $E_k=h\nu-\Phi_A+E_b$ and the photoelectron momentum magnitude |
| `alpha_scale` and `beta_scale` | Multiply the stored `alpha` and `beta` coordinates before evaluation |
| `inner_potential` | Enters the free-electron final-state expression for `kz` in hν-dependent scans; it does not affect fixed-hν in-plane conversion |

`angle_resolution` is not an input to the mapping functions. It is used only to select
an automatic interpolation grid, as described below.

## Variable experimental configurations

An experimental configuration describes the physical relation between the analyzer
slit, deflector, and sample rotation axes during acquisition. It is not a display
orientation or a generic correction for loader output.

Most endstations have one fixed configuration, which the loader assigns. Some
endstations can rotate the analyzer slit or switch between deflector mapping and a
physical sample rotation. One loader can then serve measurements acquired in different
configurations.

{meth}`xarray.DataArray.kspace.as_configuration` performs a semantic translation for
these variable-geometry setups.

| Changes | Does not change |
| --- | --- |
| The `configuration` attribute | Measured intensity values or sampling |
| Standard angle-coordinate names, translated by physical role | The physical geometry used during acquisition |
| A copy of the input object | Arbitrary or endstation-specific names from an incorrect loader |

Use {ref}`how-to-python-change-configuration` for the concrete ALS BL7 case and the
coordinate-name translation.

## Normal emission and angular offsets

The normal emission position is the measured angle pair that corresponds to emission
along the sample surface normal.

- {meth}`xarray.DataArray.kspace.set_normal` calculates the angular offsets for the
  selected configuration from this position.
- {attr}`xarray.DataArray.kspace.offsets` stores the angular offsets used for momentum
  conversion.
- Momentum conversion does not infer normal emission from the intensity maximum.
- Matrix-element asymmetry can move an intensity maximum away from the correct symmetry
  position.

The angular offsets are reference angles in the mapping functions. They are relative
to the stored angle coordinates, not an absolute position of the sample normal.

- {meth}`xarray.DataArray.kspace.set_normal` solves the offsets that map a known
  normal-emission position to zero in-plane momentum.
- Changing an angle coordinate while keeping its offset fixed changes the represented
  orientation.
- When a sample angle varies with `hv`, momentum conversion evaluates the varying
  coordinate together with its fixed reference offset. Replacing the varying
  coordinate with one constant angle changes the momentum trajectory.
- {attr}`xarray.DataArray.kspace.offsets` permits direct offset assignment when the
  sign conventions and physical reference angles are already known.

The three panels use the same simulated intensity and the same display limits. The
middle panel omits a known $\xi=3^\circ$ offset. The final panel uses that offset in the
mapping functions. The dashed lines mark $k_x=k_y=0$.

```{eval-rst}
.. plot:: explanation/momentum_conversion.py compare_angular_offsets
   :include-source: false
   :alt: Constant energy surfaces in angle coordinates calculated without and with the known xi offset
```

Use {ref}`how-to-python-change-configuration` when a variable-geometry measurement uses
a different configuration from the loader default. Use
{ref}`how-to-python-convert-angle-data` to set normal emission and perform the
conversion.

## Coordinates and interpolation

| Operation | Sampling and dimensions |
| --- | --- |
| {meth}`~erlab.accessors.kspace.MomentumAccessor.convert_coords` | Keeps measured dimensions; adds momentum coordinates that can depend on several dimensions; intensity values are unchanged unless `jacobian=True` |
| {meth}`~erlab.accessors.kspace.MomentumAccessor.convert` | Creates momentum dimensions and interpolates intensity onto a regular grid |

| Grid input | Effect |
| --- | --- |
| `bounds` | Sets the momentum limits; omitted limits are calculated from the mapped angular coordinates |
| `resolution` | Supplies a target step for each momentum axis; the final step can differ because `convert()` uses an integer number of intervals between the limits |
| Explicit `kx`, `ky`, or `kz` arrays | Supplies the exact target coordinates and overrides `bounds` and `resolution` for that axis |
| `angle_resolution` | Supplies the angular term in the automatic in-plane target-step estimate when no target step or coordinate array is given; the estimate also uses the minimum kinetic energy, the largest absolute scaled angle, and the applicable angle-scale factor |

See also {ref}`how-to-python-convert-common-grid` and {ref}`how-to-python-convert-coordinates-only`.

(explanation-kspace-jacobian)=

## Jacobian correction

Jacobian correction rescales intensity to account for the change from angular to
momentum coordinates. See {cite:t}`Gauthier2021` and {cite:t}`Miao2026` for the physical
background.

With `jacobian=True`, ERLabPy divides each measured intensity value by the absolute
forward Jacobian of its angle-to-momentum mapping:

- For a single angle-energy cut, the factor is the derivative of the slit-parallel momentum (`kx`
  or `ky`, set by the configuration) with respect to `alpha`.
- For a map across two angles, the factor is the determinant of the derivatives of (`kx`, `ky`)
  with respect to (`alpha`, `beta`).

The derivatives are calculated numerically with respect to the stored angles in degrees.
They include angle scales and offsets, with binding energy and photon energy held fixed.
The calculation is independent of source and target grid spacing. Zero, nonfinite, or
numerically unresolved factors give `NaN` intensity values.

Both conversion functions {meth}`xarray.DataArray.kspace.convert_coords` and
{meth}`xarray.DataArray.kspace.convert` applies the correction at the measured points
when `jacobian=True` is set.

For hν-dependent scans, the same angular correction is applied separately at each photon
energy. It does not include an additional $h\nu$-to-$k_z$ Jacobian factor. This is
because photon-energy scans are often accompanied by intensity changes that include beam
intensity variations, high-order contamination, cross-section changes, and other effects
that are not included in the simple free-electron final state model.

## hν–dependent scans

An hν-dependent scan changes the photoelectron kinetic energy. Momentum conversion
uses the photon energy to calculate the out-of-plane momentum. For $h\nu$–dependent
scans:

- `eV` must already contain binding energy for hν-dependent data.
- `inner_potential` is a parameter of the free-electron final-state approximation, not
  an analyzer setting.
- A sample angle that varies with `hv` produces a curved path through momentum space.
- {meth}`~erlab.accessors.kspace.MomentumAccessor.hv_to_kz` returns calculated
  coordinates. It does not add measurements at new photon energies.

Use {ref}`how-to-python-convert-photon-energy-scan` for conversion. Use
{ref}`how-to-plotting-photon-energy-annotations` for calculated paths on converted
momentum-space intensity.
