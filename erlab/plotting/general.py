"""General plotting utilities."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import AxesWidget
from pyimagetool import RegularDataArray, imagetool

__all__ = ['ximagetool','proportional_colorbar','LabeledCursor']

def proportional_colorbar(mappable=None, cax=None, ax=None, **kwargs):
    r"""Replaces the current colorbar or creates a new colorbar with 
    proportional spacing.

    The default behavior of colorbars in `matplotlib` does not support
    colors proportional to data in different norms. This function
    circumvents this behavior. 

    Parameters
    ----------
    mappable : `matplotlib.cm.ScalarMappable`, optional
        The `matplotlib.cm.ScalarMappable` described by this colorbar.

    cax : `matplotlib.axes.Axes`, optional
        Axes into which the colorbar will be drawn.

    ax : `matplotlib.axes.Axes`, list of Axes, optional
        One or more parent axes from which space for a new colorbar axes
        will be stolen, if `cax` is None.  This has no effect if `cax`
        is set. If `mappable` is None and `ax` is given with more than
        one Axes, the function will try to get the mappable from the
        first one.

    **kwargs : dict, optional
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the 
        `matplotlib` documentation for a list of all possible arguments.

    Returns
    -------
    cbar : `matplotlib.colorbar.Colorbar`

    Examples
    --------
    ::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        # Create example data and plot
        X, Y = np.mgrid[0:3:complex(0, 100), 0:2:complex(0, 100)]
        pcm = plt.pcolormesh(X, Y, (1 + np.sin(Y * 10.)) * X**2,
                             norm=colors.PowerNorm(gamma=0.5),
                             cmap='Blues_r', shading='auto')

        # Plot evenly spaced colorbar
        proportional_colorbar()

    """
    if cax is None:
        if ax is None:
            ax = plt.gca()
            ax_ref = ax
        else:
            try: 
                ax_ref = ax.flatten()[0]
            except AttributeError:
                ax_ref = ax
    else:
        ax_ref = cax
    if mappable is None:
        mappable = plt.gci()
        if mappable is None:
            try:
                mappable = ax_ref.collections[-1]
            except IndexError:
                pass
            if mappable is None:
                raise RuntimeError('No mappable was found to use for colorbar '
                                   'creation. First define a mappable such as '
                                   'an image (with imshow) or a contour set ('
                                   'with contourf).')
    if mappable.colorbar is None:
        plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)
    ticks = mappable.colorbar.get_ticks()
    mappable.colorbar.remove()
    kwargs.setdefault('ticks',ticks)
    cbar = plt.colorbar(
        mappable=mappable,
        cax=cax, ax=ax,
        spacing='proportional',
        boundaries=mappable.norm.inverse(np.linspace(0,1,mappable.cmap.N)),
        **kwargs,
    )
    return cbar

class LabeledCursor(AxesWidget):
    """
    A crosshair cursor that spans the axes and moves with mouse cursor.
    For the cursor to remain responsive you must keep a reference to it.
    Unlike `matplotlib.widgets.Cursor`, this also shows the current
    cursor location.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    textOn : bool, default: True
        Whether to show current cursor location.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.
    textprops : dict, default: {}
        Keyword arguments to pass onto the text object.
    
    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.
    """

    def __init__(self, ax, horizOn=True, vertOn=True, textOn=True,
                 useblit=True, textprops={}, **lineprops):
        super().__init__(ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.textOn = textOn
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops['animated'] = True
            textprops['animated'] = True

        lcolor = lineprops.pop("color", lineprops.pop("c", "k"))
        ls = lineprops.pop("ls", lineprops.pop("linestyle", "--"))
        lw = lineprops.pop("lw", lineprops.pop("linewidth", 0.8))
        lineprops.update(dict(color=lcolor, ls=ls, lw=lw, visible=False))
        
        tcolor = textprops.pop("color", textprops.pop("c", lcolor))
        textprops.update(dict(color=tcolor, visible=False,
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform=ax.transAxes))

        self.lineh = ax.axhline(ax.get_ybound()[0], **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], **lineprops)
        with plt.rc_context({'text.usetex':False}):
            self.label = ax.text(0.95, 0.95, '', **textprops)
        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)
        self.label.set_visible(False)

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)
            self.label.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.label.set_text('(%1.3f, %1.3f)' % (event.xdata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        self.label.set_visible(self.visible and self.textOn)

        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.ax.draw_artist(self.label)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

def ximagetool(da:xr.DataArray, prebin=None, cmap='viridis'):
    """
    Open imagetool from xarray.DataArray. Note that NaN values are displayed as zero.
    """
    if isinstance(da, xr.Dataset):
        da = da.spectrum
    if prebin is not None:
        da = da.coarsen(prebin).mean()
    dims = da.dims
    if set(dims) == set(('kx', 'ky', 'eV')):
        da = da.transpose('kx', 'ky', 'eV')

    try:
        units = ['('+da[dims[i]].units+')' for i in range(len(dims))]
    except AttributeError:
        units = ['' for i in range(len(dims))]
    data = RegularDataArray(
        da.fillna(0),
        dims=[dims[i]+' '+units[i] for i in range(len(dims))]
    )

    # TODO: implement 4dim and raise error
    tool = imagetool(data)
    # tool.set_all_cmaps(cmap)
    # tool.pg_win.load_ct('viridis')
    # tool.pg_win.update()
    # tool.info_bar.cmap_combobox.currentTextChanged.connect(tool.set_all_cmaps)
    
    return tool
