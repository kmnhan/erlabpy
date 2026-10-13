(guide-angle-calculator)=

# Angle calculator

Interactive calculator for choosing angles and photon energies to reach a target region
of momentum space.

Open {menuselection}`Apps --> Angle calculator` in ImageTool Manager, or use
{func}`erlab.interactive.angle_calculator`:

```python
import erlab.interactive as eri

eri.angle_calculator()
```

It can also be started from the terminal with `python -m erlab.interactive.angle_calculator`.

## Geometry

In the {guilabel}`Geometry` tab, you can set the experimental configuration, sample
orientation, and photon energy to match the intended measurement.

Adjust the $\alpha$ and $\beta$ ranges to contain the region of interest. You can drag
the line that represents the current cut in either view to choose the $\beta$ at which
the slit crosses a target feature. The meaning of $\beta$ depends on the Configuration:

| Configuration | $\beta$ | Fixed sample angles |
| --- | --- | --- |
| Deflector | Deflector angle | Polar and tilt |
| Vertical slit without deflectors | Polar angle | Tilt |
| Horizontal slit without deflectors | Tilt angle | Polar |

For the momentum conversion conventions, see {doc}`../../../explanation/momentum-conversion`.

## Brillouin zones

In the {guilabel}`Lattice` tab, enter the conventional cell parameters, select the
centering, and set the lattice rotation.

The resulting curves represent intersections of the Brillouin zone boundaries with the
calculated constant energy surface. They therefore change with photon energy, binding
energy, and inner potential, and thus do not resemble a section at fixed $k_z$.

## Photon energy coverage

Enable {guilabel}`Show α–kz plot` to inspect measurements at different photon energies for the current $\beta$. Drag the current-{guilabel}`hν` curve to change the photon energy.

(angle-calculator-overlays)=

## Overlays

Use {guilabel}`Overlays` to place ellipses that represents pockets or a reference image
in $k_x$ and $k_y$ coordinates. The angle view will update to show the corresponding
features in angle space.

(angle-calculator-pockets)=

### Pockets

Select {guilabel}`Add` to add a pocket, then drag its center, radius, or rotation
handles to adjust its size. You can draw symmetry-related copies about the momentum
origin by enabling {guilabel}`Apply point group`. For groups with mirrors,
{guilabel}`Mirror axis` sets the reflection direction in degrees. The Angle
calculator also repeats pockets in the visible reciprocal cells, including when
{guilabel}`Show Brillouin zones` is off. Note that $k_z$ dispersion is ignored.

In the pocket list, you can use {guilabel}`Copy` and {guilabel}`Paste` in the pocket
list to transfer selected pockets between calculators or ktool windows.

(angle-calculator-reference-image)=

### Reference image

Open a PNG or JPEG image that shows band contours in momentum space and set its center,
{guilabel}`Image width` in Å⁻¹, and rotation in momentum space. Scaling preserves the
aspect ratio. Reduce {guilabel}`Opacity` to see the data underneath. After alignment,
enable {guilabel}`Lock image` to prevent accidental dragging.

(angle-calculator-laser-mode)=

## Laser ARPES mode

Enable {guilabel}`Laser ARPES mode` in {menuselection}`Preferences --> Angle calculator`
to expose the sample work function and two optional controls in new windows:

- {guilabel}`Show sample escape limit` draws the maximum in-plane momentum allowed by
  the sample kinetic energy.
- {guilabel}`Parallel-plate correction` changes the angle-to-momentum conversion to
  include the effective bias due to the the sample–analyzer work function difference.

(angle-calculator-parallel-plate)=

### Parallel-plate correction

The correction assumes zero applied bias, a uniform field between parallel sample and
analyzer planes, and a *position-limit* lens response, where the reported angle depends
on the electron's entry position {cite:p}`Gauthier2021,Miao2026`. Use it when these
assumptions seem appropriate.

The sample normal must align with the analyzer axis. Azimuth rotation and deflector
angles are allowed, but polar and tilt must be zero. Incompatible values will clear and
disable the correction.

## Saved settings

Use {menuselection}`Edit --> Undo` and {menuselection}`Edit --> Redo` for geometry,
lattice, and overlay edits. A completed drag is one Undo step.

Manager retains the calculator when its window is closed. Saving the workspace
stores its settings, pockets, and embedded reference image; the original image file
is not needed on restore. New windows use the defaults in the
{guilabel}`Angle calculator` settings category and the saved ktool lattice settings.

The {guilabel}`Laser ARPES mode` preference controls visibility of the laser controls.
Hiding them leaves active settings in effect and shows a read-only summary.
