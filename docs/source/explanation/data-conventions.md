(data-conventions)=

# ARPES data conventions

ERLabPy loaders translate endstation-specific files and metadata into labeled xarray
objects. ERLabPy's ARPES-specific tools use a common set of coordinate and attribute
names. These names are not a universal file-format standard.

Generic xarray operations do not depend on these names. Momentum conversion,
Fermi edge analysis, and ARPES-aware plotting use them to find physical coordinates
and experimental metadata without endstation-specific code.

## Coordinates and metadata

A dimension coordinate gives the physical values along an array axis. A scalar
coordinate records a fixed measurement condition. An attribute stores metadata that
applies to the complete object.

| Name | Storage | Unit | Role |
| --- | --- | --- | --- |
| `eV` | Coordinate | eV | Binding or kinetic energy |
| `alpha` | Coordinate | deg | Analyzer angle |
| `beta` | Coordinate | deg | Mapping or deflector angle |
| `delta` | Coordinate | deg | Sample azimuth |
| `xi` | Coordinate | deg | Sample orientation angle |
| `chi` | Coordinate | deg | Polar angle in deflector configurations |
| `hv` | Coordinate | eV | Photon energy, either scalar or scanned |
| `sample_temp` | Coordinate or attribute | K | Sample temperature |
| `configuration` | Attribute | — | Experimental geometry, stored as 1–4 |
| `sample_workfunction` | Attribute | eV | Work function used for the kinetic-energy scale |
| `angle_resolution` | Attribute | deg | Angular input to the default in-plane target-step estimate |
| `inner_potential` | Attribute | eV | Free-electron final-state parameter for out-of-plane momentum |
| `alpha_scale`, `beta_scale` | Attributes | — | Optional compensation for known angle coordinate scale errors |

Attributes do not participate in xarray alignment. Verify the geometry attributes
after concatenation, arithmetic, or another operation that returns a new object.

## Energy coordinates

ERLabPy uses one name, `eV`, for binding and kinetic energy. Momentum conversion applies
these rules:

| `eV` and `hv` coordinates | Interpretation |
| --- | --- |
| `eV` contains zero or negative values | Binding energy; occupied states use negative values |
| Nonscalar `eV` is all positive and `hv` is fixed | Kinetic energy; ERLabPy converts it to binding energy |
| `hv` contains a scan | `eV` must already be binding energy |

The kinetic-energy conversion uses `hv` and `sample_workfunction`. Incorrect values can
produce a wrong momentum scale or a nonphysical kinetic energy.

(nomenclature)=

## Experimental geometry

Momentum conversion follows the four configurations defined in {cite:t}`ishida2018kconv`.
`alpha` is always the analyzer angle. `delta` is always the sample azimuth. The roles of
the other angles depend on `configuration`.

| `configuration` | Geometry | Role of `beta` | Polar angle | Tilt angle | Slit momentum axis |
| --- | --- | --- | --- | --- | --- |
| 1 | Type 1 | Polar mapping | `beta` | `xi` | `kx` |
| 2 | Type 2 | Tilt mapping | `xi` | `beta` | `ky` |
| 3 | Type 1 with deflector | Deflector | `chi` | `xi` | `kx` |
| 4 | Type 2 with deflector | Deflector | `chi` | `xi` | `ky` |

For a two-dimensional analyzer in a deflector configuration, ERLabPy stores the
deflector angle as `beta`.

The configuration controls angle roles and momentum-axis assignment. It is not only a
descriptive label. If a loader assigns the wrong configuration, use the measured
acquisition geometry to correct it before you set normal emission or convert to
momentum space.

## Metadata defaults

The momentum accessor has fallback values for some metadata. It uses these values when
the metadata is absent. They do not replace measured or justified values.

| Missing input | ERLabPy behavior | Scientific action |
| --- | --- | --- |
| `configuration` | Raises an incomplete-data error | Recover the acquisition geometry |
| `sample_workfunction` | Warns and uses 4.5 eV | Set the electrically connected system work function |
| `angle_resolution` | Uses 0.1° to estimate the default in-plane target step | Set the measured angular resolution before using the automatic interpolation grid |
| `inner_potential` | Warns and uses 10 eV | Set a value justified by the out-of-plane analysis |
| `alpha_scale`, `beta_scale` | Uses 1.0 | Leave unset unless the angle coordinates have a known scale error |

## Related procedures

- Use the {doc}`loading and saving guides <../how-to/python/loading-and-saving>` to load
  supported data and inspect coordinates and metadata that use the ERLabPy names.
- Use the {doc}`momentum-conversion guides <../how-to/python/momentum-conversion>` to
  correct a configuration, set normal emission, and convert data.
- Read {doc}`momentum conversion <momentum-conversion>` for the distinction between the
  momentum conversion functions and grid interpolation.
- Use {class}`erlab.constants.AxesConfiguration` and
  {class}`xarray.DataArray.kspace <erlab.accessors.kspace.MomentumAccessor>` for exact
  values and accessor behavior.
