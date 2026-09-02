(how-to-gui-analyze-roi-in-imagetool)=

(imagetool-roi)=

# Data inspection and selection

Use these guides to compare linked views and extract a region of interest.

(how-to-gui-compare-linked-data)=

## Comparing data in linked ImageTools

Use linked ImageTools when several datasets share coordinates and must be inspected at
the same positions.

From Python, open the arrays together:

```python
import erlab.interactive as eri

eri.itool([data_a, data_b], link=True)
```

In Manager, select the ImageTool rows and choose {guilabel}`Link` or press
{kbd}`Ctrl+L`. Move a cursor and change a bin width to confirm that the windows follow
the same coordinates.

Linked windows share cursor positions, bin widths, cursor counts, and plot-layout
proportions. If the coordinates are not compatible, select only comparable windows or
inspect them independently. Use {guilabel}`Unlink` or {kbd}`Ctrl+Shift+L` when later
changes must remain independent.

(how-to-gui-extract-polygon-path)=

## Extracting data along a polygonal path

Use this task when a cut must follow a path that is not aligned with the data axes.

1. Right-click an image plot and choose {guilabel}`Add Polygon ROI`.
2. Drag the handles to place the vertices along the required path.
3. Click a segment to add a vertex when the path needs another point.
4. Right-click the ROI and choose {guilabel}`Edit ROI…` when you must enter exact
   coordinates.
5. Leave {guilabel}`Closed` off.
6. Right-click the ROI and choose {guilabel}`Slice Along ROI Path`.
7. Set the step size and the name of the new path dimension.
8. Select the required {guilabel}`Result Placement` and create the result.
9. Inspect the path coordinate and the source coordinates in the new ImageTool.

The operation uses {func}`erlab.analysis.interpolate.slice_along_path`. It interpolates
the complete data volume along the path. A smaller step adds output samples, but it does
not add measured resolution.

(how-to-gui-mask-polygon)=

## Masking data with a polygonal ROI

Use this task when an analysis must retain values on one side of a polygon boundary.

1. Right-click an image plot and choose {guilabel}`Add Polygon ROI`.
2. Drag the handles to place the polygon vertices.
3. Click a segment to add a vertex when the boundary needs another point.
4. Right-click the ROI and choose {guilabel}`Edit ROI…` when you must enter exact
   coordinates.
5. Turn on {guilabel}`Closed`.
6. Right-click the ROI and choose {guilabel}`Mask Data with ROI`.
7. Select whether to invert the mask.
8. Select whether to trim coordinate ranges that contain only masked values.
9. Select the required {guilabel}`Result Placement` and create the result.
10. Inspect the masked region and coordinate ranges in the new ImageTool.

The operation uses {func}`erlab.analysis.mask.mask_with_polygon`. It applies the polygon
to the complete data volume, not only to the visible slice. Masked points remain missing
values unless you trim ranges that contain no retained data.
