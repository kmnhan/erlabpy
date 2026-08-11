# Momentum conversion and normal emission

Calibrate the energy axis before this workflow. Keep normal-emission choices and all
geometry values in visible notebook cells.

## Contents

- [Check prerequisites](#check-prerequisites)
- [Group scans by sample alignment](#group-scans-by-sample-alignment)
- [Use an existing calibration](#use-an-existing-calibration)
- [Prepare visual diagnostics](#prepare-visual-diagnostics)
- [Compare photon energies safely](#compare-photon-energies-safely)
- [Infer normal emission visually](#infer-normal-emission-visually)
- [Recognize underdetermined data](#recognize-underdetermined-data)
- [Apply and validate the conversion](#apply-and-validate-the-conversion)
- [Use KTool for user confirmation](#use-ktool-for-user-confirmation)

## Check prerequisites

Inspect:

- the experimental `configuration`;
- swept and scalar angle coordinates;
- photon energy, work function, and inner potential when applicable;
- angle-coordinate units and monotonicity;
- angle-scale compensation factors;
- explicit offset attributes and any reference dataset;
- whether the measured angular range includes normal emission.

Compare each scalar angle coordinate with the raw analyzer attribute and the
experimental log. A loader can emit a default scalar zero when the acquisition used a
nonzero fixed angle. Restore the authoritative value before conversion:

```python
fixed_beta = float(experimental_log_row["DA"])
data_with_geometry = energy_corrected_data.assign_coords(beta=fixed_beta)
```

Use the user's stated metadata authority. Record the original coordinate, replacement
value, and source. Do not replace a swept coordinate with one scalar value.

`data.kspace.offsets` returns zero for missing offset attributes. Check the attributes
before treating zero as a calibration:

```python
offsets = dict(data.kspace.offsets)
stored_offset_keys = {
    key for key in offsets if f"{key}_offset" in data.attrs
}
```

An empty `stored_offset_keys` set means that the displayed zeros are defaults.

## Group scans by sample alignment

Build an alignment table before selecting a normal-emission calibration. Include:

- the sample, cleavage, and termination;
- file identifier and acquisition order;
- photon energy and polarization;
- sample-position and sample-angle motor values;
- analyzer lens, slit, pass energy, and acquisition mode;
- log notes that report a realignment, sample motion, or geometry change.

Use one `(alpha_normal, beta_normal)` pair for every photon energy in one unchanged
sample alignment. Normal emission does not change when only photon energy changes. Do
not fit one offset pair per photon energy.

Split the alignment group when the sample was moved or realigned. A shared sample name
or similar motor readbacks do not prove that two acquisitions have the same alignment.
Prefer a reference map
acquired immediately before or after the scan series. Do not use a later map when the
log says that the angle changed in between.

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

Search saved KTool code, a saved workspace, or the notebook for an approved `set_normal`
operation before inferring new values. Report that source directly. Do not describe a
recovered human calibration as an agent visual deduction. A cursor position alone is not
an approved calibration unless its purpose is documented or the user confirms it.

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

## Prepare visual diagnostics

Matrix elements can make one side of a band much brighter than the other. Determine
the center from band positions and contour geometry, not from intensity maxima.

Prepare all of these views before proposing a center:

1. Keep a raw, unsymmetrized angle-space view.
2. Average a finite energy window that is wide enough for the count rate and energy
   resolution. Record the full width.
3. Apply coordinate-aware smoothing in physical angle and energy units. Do not apply a
   derivative to a visibly noisy image.
4. Add a normalized or local-contrast view to reduce broad matrix-element modulation.
5. Add a derivative, minimum-gradient, or curvature view only after averaging and
   smoothing.
6. Show the raw and processed views together. A processed view cannot be the only
   evidence.

Use coordinate-aware ERLabPy smoothing:

```python
diagnostic_map = energy_corrected_data.qsel(
    eV=diagnostic_energy,
    eV_width=diagnostic_energy_width,
)
smoothed_map = era.image.gaussian_filter(
    diagnostic_map,
    sigma={"alpha": alpha_smoothing, "beta": beta_smoothing},
)
```

Choose smoothing widths from the coordinate steps and feature widths. Do not copy one
numeric width between instruments without inspection.

Find the analyzer acceptance mask before normalization or differentiation. Crop the
boundary or erode the valid-data mask by more than the smoothing kernel radius. Do not
use a circular detector or deflector boundary as a symmetry feature. Its center is set
by analyzer acceptance and is unrelated to sample normal emission.

## Compare photon energies safely

For one unchanged sample alignment, a feature at fixed in-plane momentum changes its
emission angle as photon energy changes. Corresponding contours can contract or expand
about the common normal-emission angle. Use this as supporting geometry evidence.

Compare only a feature that can be identified as the same branch or contour at both
photon energies. Use energy averaging, smoothing, and a raw-data check for each view.
Bulk bands can move with out-of-plane momentum. Matrix elements can hide one band and
reveal another. Therefore:

- do not correlate complete images as if all bands were unchanged;
- do not combine all derivative ridges into one alignment score;
- do not join peaks only because they form a smooth numerical path;
- do not force a match through a photon energy where the feature disappears;
- show the selected corresponding feature and rejected alternatives.

If no corresponding feature remains identifiable, mark the comparison as
underdetermined and use the interactive user-confirmation workflow.

## Infer normal emission visually

Use visual inference to propose a KTool starting position when no approved calibration
exists and the data contains enough physical information. Do not treat agent confidence
as approval for publication or quantitative momentum conversion.

1. Plot several raw angle-space constant-energy slices or representative projections.
2. Use the same axis directions, color normalization, and aspect for all slices.
3. Save the diagnostic figure and inspect the rendered pixels with the available image
   viewer.
4. Identify a tentative center from a known zone-center feature, a consistent contour
   center, or a cut known to pass through normal emission. Do not use apparent mirror
   brightness or an intensity maximum.
5. Test a small grid of nearby `(alpha, beta)` candidates. Use coordinate step sizes to
   define the grid rather than arbitrary precision.
6. Convert unsymmetrized previews for several energies.
7. Draw zero-momentum guides. Add a Brillouin-zone overlay when verified lattice vectors
   and orientation are available.
8. Use the supported center only as the initial KTool value. Obtain user confirmation
   before a publication or quantitative conversion.

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
visual review, but it must not override contradictory raw data or known geometry. Do
not accept a score that depends on the analyzer mask, unrelated photon-energy bands, or
noisy derivative contours.

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

For publication or quantitative momentum conversion, also request user confirmation
when no approved calibration exists, even if the agent sees only one candidate.

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

## Use KTool for user confirmation

Use KTool as the acceptance step when no approved normal-emission calibration exists.
Use ImageTool first when linked raw and processed views help the user select the region:

```python
import erlab.interactive as eri

eri.itool(
    [raw_diagnostic_map, processed_diagnostic_map],
    link=True,
    link_colors=False,
)
```

Open KTool with the energy-corrected, unsymmetrized source data. A static estimate can
seed the controls:

```python
import erlab.interactive as eri

eri.ktool(
    energy_corrected_data,
    avec=real_space_lattice_vectors,
    initial_normal_emission=(alpha_candidate, beta_candidate),
)
```

When a standalone process must keep the window open, pass `execute=True`. In a notebook
with an active Qt event loop, use the default or `execute=False` so the user can continue
to interact with the notebook.

Tell the user to:

1. inspect several energy slices and integration widths;
2. adjust `alpha` and `beta` normal emission;
3. ignore the circular analyzer-acceptance boundary;
4. use a Brillouin-zone overlay only with verified lattice vectors and orientation;
5. report the displayed angles or use **Copy to clipboard** and return the generated
   code.

Keep the tool open and wait for the user. Do not continue to final momentum conversion
while the selection is pending. If interactive display is unavailable, save the raw
and processed diagnostics and ask the user for the accepted values.

After confirmation, copy the accepted public code into the notebook. Record the values
as `user-selected in KTool`, include the source file and alignment group, close the GUI,
and rerun the conversion from a clean kernel. Do not make the notebook depend on an
open tool, clipboard state, or saved workspace.
