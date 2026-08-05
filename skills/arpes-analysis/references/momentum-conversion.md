# Momentum conversion and normal emission

Calibrate the energy axis before this workflow. Keep normal-emission choices and all
geometry values in visible notebook cells.

## Contents

- [Check prerequisites](#check-prerequisites)
- [Use an existing calibration](#use-an-existing-calibration)
- [Infer normal emission visually](#infer-normal-emission-visually)
- [Recognize underdetermined data](#recognize-underdetermined-data)
- [Apply and validate the conversion](#apply-and-validate-the-conversion)
- [Use ktool without losing reproducibility](#use-ktool-without-losing-reproducibility)

## Check prerequisites

Inspect:

- the experimental `configuration`;
- swept and scalar angle coordinates;
- photon energy, work function, and inner potential when applicable;
- angle-coordinate units and monotonicity;
- angle-scale compensation factors;
- explicit offset attributes and any reference dataset;
- whether the measured angular range includes normal emission.

`data.kspace.offsets` returns zero for missing offset attributes. Check the attributes
before treating zero as a calibration:

```python
offsets = dict(data.kspace.offsets)
stored_offset_keys = {
    key for key in offsets if f"{key}_offset" in data.attrs
}
```

An empty `stored_offset_keys` set means that the displayed zeros are defaults.

## Use an existing calibration

Prefer a user-approved calibration acquired in a compatible geometry. Transfer its
normal-emission angles, azimuth, and angle-scale factors with the public accessor:

```python
data_for_conversion = energy_corrected_data.copy(deep=False)
data_for_conversion.attrs = energy_corrected_data.attrs.copy()
data_for_conversion.kspace.set_normal_like(reference_data)
```

Verify that the reference and target use compatible configurations and angle meanings.
Do not copy a raw offset dictionary between incompatible configurations.

When the normal-emission angles are already approved, set them directly:

```python
alpha_normal = 1.2
beta_normal = -0.4
azimuth_offset = 30.0

data_for_conversion = energy_corrected_data.copy(deep=False)
data_for_conversion.attrs = energy_corrected_data.attrs.copy()
data_for_conversion.kspace.set_normal(
    alpha_normal,
    beta_normal,
    delta=azimuth_offset,
)
```

Use `set_normal`. Do not derive offset signs by hand.

## Infer normal emission visually

Use visual inference only when no approved calibration exists and the data contains
enough physical information.

1. Plot several raw angle-space constant-energy slices or representative projections.
2. Use the same axis directions, color normalization, and aspect for all slices.
3. Save the diagnostic figure and inspect the rendered pixels with the available image
   viewer.
4. Identify a tentative center from a known zone-center feature, a consistent mirror or
   rotational center, or a cut known to pass through normal emission.
5. Test a small grid of nearby `(alpha, beta)` candidates. Use coordinate step sizes to
   define the grid rather than arbitrary precision.
6. Convert unsymmetrized previews for several energies.
7. Draw zero-momentum guides. Add a Brillouin-zone overlay when verified lattice vectors
   and orientation are available.
8. Accept the center only when several raw and converted views support the same choice.

If the agent cannot inspect rendered images, save the diagnostics and request user
review. Do not replace visual evidence with an unverified numerical guess.

Make a shallow data copy with independent attributes for each candidate. This avoids
copying a large array and prevents one candidate from changing another candidate's
metadata:

```python
candidate_data = energy_corrected_data.copy(deep=False)
candidate_data.attrs = energy_corrected_data.attrs.copy()
candidate_data.kspace.set_normal(alpha_candidate, beta_candidate, delta=delta_candidate)
candidate_map = candidate_data.kspace.convert().qsel(
    eV=preview_energy,
    eV_width=preview_energy_width,
)
```

Do not use an n-fold-symmetrized image as evidence for the center. Symmetrization can
make an incorrect center look plausible. A quantitative symmetry score can support the
visual review, but it must not override contradictory raw data or known geometry.

Record the accepted values and evidence in a markdown cell next to the diagnostics:

- `alpha_normal` and `beta_normal` in raw data coordinates;
- azimuth and angle-scale values;
- the source of the estimate;
- which energies and features supported it;
- whether a reference or user confirmed it;
- the remaining uncertainty.

## Recognize underdetermined data

Request user input instead of selecting a value when:

- an angle-energy cut provides a swept `alpha` coordinate but no evidence for its fixed
  orthogonal `beta` angle;
- the measured map does not include normal emission;
- several symmetry centers are plausible;
- matrix-element asymmetry hides the expected center;
- magnetic or other symmetry-breaking behavior invalidates the assumed symmetry;
- configuration, work function, photon energy, or required scalar angles are missing;
- the Brillouin-zone orientation is unknown and materially changes the conclusion.

Retain the diagnostic cells and candidate figure. Show the user the proposed values and
the source of the ambiguity. Resume the notebook after confirmation.

## Apply and validate the conversion

Apply the accepted values to a new variable, then convert:

```python
data_for_conversion = energy_corrected_data.copy(deep=False)
data_for_conversion.attrs = energy_corrected_data.attrs.copy()
data_for_conversion.kspace.set_normal(
    alpha_normal,
    beta_normal,
    delta=azimuth_offset,
)
data_for_conversion.kspace.work_function = work_function

momentum_data = data_for_conversion.kspace.convert(
    bounds=momentum_bounds,
    resolution=momentum_resolution,
)
```

Let ERLabPy infer bounds and resolution for the first diagnostic conversion. Supply them
explicitly for the final notebook when consistent grids matter across datasets.

Validate:

- the converted zero lies at the accepted normal-emission position;
- several energies give a consistent center and orientation;
- the converted range and resolution are plausible for the kinetic energy;
- expected symmetry points agree with a verified Brillouin-zone overlay;
- no coordinate was silently dropped or transposed;
- the final plotted data is unsymmetrized unless symmetrization is a separate requested
  analysis step.

For an in-plane map with reciprocal lattice vectors `bvec`, overlay repeated zones on
the existing plot:

```python
kx_min, kx_max = sorted(ax.get_xlim())
ky_min, ky_max = sorted(ax.get_ylim())
eplt.plot_in_plane_bz(
    bvec,
    bounds=(kx_min, kx_max, ky_min, ky_max),
    ax=ax,
    color="w",
    lw=0.75,
)
```

## Use ktool without losing reproducibility

Use `ktool` when interactive adjustment is more effective:

```python
import erlab.interactive as eri

eri.ktool(
    energy_corrected_data,
    avec=real_space_lattice_vectors,
    initial_normal_emission=(alpha_candidate, beta_candidate),
)
```

Inspect the unsymmetrized preview. Use Brillouin-zone and high-symmetry overlays only
when their lattice inputs are verified. Copy the accepted generated code back into the
notebook and rerun it without depending on the open tool or clipboard.
