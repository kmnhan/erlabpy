(data-conventions)=

# ARPES data conventions

ERLabPy loaders translate endstation-specific files and metadata into labeled xarray
objects. ERLabPy's ARPES-specific tools use a common set of coordinate and attribute
names.

These names are used by some analysis routines, including Momentum conversion, Fermi
edge fitting, and ARPES-aware plotting in order to find physical coordinates and
experimental metadata without endstation-specific code. Other analysis routines work
regardless of the coordinate names.

## Coordinates and metadata

- A dimension coordinate gives the physical values along an array axis.
- A scalar coordinate records a measurement condition that does not vary along any axis.
- A non-scalar coordinate that is not a dimension coordinate represents a physical
  quantity that varies along one or more axes but is not used for indexing (for example,
  temperature recorded during a scan along the time axis).
- Everything else can be stored as an attribute.

| Name | Storage | Unit | Role |
| --- | --- | --- | --- |
| `eV` | Coordinate | eV | Binding or kinetic energy |
| `alpha`, `beta`, `delta`,`xi`, `chi` | Coordinate | deg | See {ref}`nomenclature` |
| `hv` | Coordinate | eV | Photon energy |
| `sample_temp` | Coordinate or attribute | K | Sample temperature |
| `configuration` | Attribute | — | Experimental geometry, stored as 1–4 |
| `sample_workfunction` | Attribute | eV | System work function used to convert binding energy to kinetic energy |
| `angle_resolution` | Attribute | deg | Angular resolution for the default momentum step estimate |
| `inner_potential` | Attribute | eV | Inner potential for out-of-plane momentum conversion |
| `alpha_scale`, `beta_scale` | Attributes | — | Optional compensation for known angle coordinate scale errors |

## Energy coordinates

ERLabPy uses one name, `eV`, for both binding and kinetic energy. Momentum conversion
applies these rules:

| `eV` and `hv` coordinates | Interpretation |
| --- | --- |
| `eV` contains zero or negative values | Binding energy; occupied states use negative values |
| Nonscalar `eV` is all positive and `hv` is fixed | Kinetic energy; ERLabPy converts it to binding energy |
| Multiple values for `hv` | `eV` must already be in binding energy |

Incorrect values can produce a wrong momentum scale or a nonphysical kinetic energy.

(nomenclature)=

## Experimental geometry

Momentum conversion follows the four configurations defined in {cite:t}`ishida2018kconv`.
`alpha` is always the analyzer angle. `delta` is always the sample azimuth. The roles of
the other angles depend on `configuration`.

The following table summarizes angle conventions for commonly encountered configurations
with a vertical cryostat.

```{eval-rst}
+-------------------+---------------------------+---------------+-----------+----------+-----------+-----------+-----------+
| Configuration     | Analyzer slit orientation | Mapping angle | Polar     | Tilt     | Deflector | Azimuth   | Analyzer  |
+===================+===========================+===============+===========+==========+===========+===========+===========+
| 1 (Type 1)        | Vertical                  | Polar         | ``beta``  | ``xi``   | –         | ``delta`` | ``alpha`` |
+-------------------+---------------------------+---------------+-----------+----------+-----------+           |           |
| 2 (Type 2)        | Horizontal                | Tilt          | ``xi``    | ``beta`` | –         |           |           |
+-------------------+---------------------------+---------------+-----------+----------+-----------+           |           |
| 3 (Type 1 + DA)   | Vertical                  | Deflector     | ``chi``   | ``xi``   | ``beta``  |           |           |
+-------------------+---------------------------+               +           |          |           |           |           |
| 4 (Type 2 + DA)   | Horizontal                |               |           |          |           |           |           |
+-------------------+---------------------------+---------------+-----------+----------+-----------+-----------+-----------+
```

For instance, imagine a typical Type 1 setup with a vertical analyzer slit that acquires
maps by rotating about the `z` axis in the lab frame. In this case, the polar angle
(rotation about `z`) is $\beta$, and the tilt angle becomes $\xi$.

:::{note}

Analyzers that measure two-dimensional angular information, such as time-of-flight
analyzers, can be treated as hemispherical analyzers with a deflector.

:::

## Related procedures

- Use {doc}`../how-to/python/loading-and-saving` to load supported data and inspect
  coordinates and metadata that use the ERLabPy names.
- Use {doc}`../how-to/python/momentum-conversion` to select a variable experimental
  configuration, set normal emission, and convert data.
- Read {doc}`momentum-conversion` for the distinction between the momentum conversion
  functions and grid interpolation.
- Use {class}`erlab.constants.AxesConfiguration` and
  {class}`xarray.DataArray.kspace <erlab.accessors.kspace.MomentumAccessor>` for exact
  values and accessor behavior.
