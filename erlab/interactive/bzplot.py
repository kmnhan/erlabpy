import sys

import numpy as np
import numpy.typing as npt

import erlab.plotting.erplot as eplt
from erlab.interactive.utilities import ParameterGroup

from qtpy import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import mpl_toolkits


def _ang(v1, v2):
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def abc2avec(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    sa, ca, sb, cb, cg = (
        np.sin(alpha),
        np.cos(alpha),
        np.sin(beta),
        np.cos(beta),
        np.cos(gamma),
    )

    gp = np.arccos(np.clip((ca * cb - cg) / (sa * sb), -1.0, 1.0))
    cgp, sgp = np.cos(gp), np.sin(gp)
    return np.array(
        [
            [a * sb, 0, a * cb],
            [-b * sa * cgp, b * sa * sgp, b * ca],
            [0, 0, c],
        ]
    )


def avec2abc(avec):
    a, b, c = [np.linalg.norm(x) for x in avec]
    alpha = _ang(avec[1] / b, avec[2] / c)
    beta = _ang(avec[2] / c, avec[0] / a)
    gamma = _ang(avec[0] / a, avec[1] / b)
    return a, b, c, alpha, beta, gamma


def to_reciprocal(avec):
    return 2 * np.pi * np.linalg.inv(avec).T


def to_real(bvec):
    return np.linalg.inv(bvec.T / 2 / np.pi)


class BZPlotter(QtWidgets.QMainWindow):
    def __init__(
        self,
        params: tuple[float, ...] | npt.NDArray[np.float64] | None = None,
        type="bvec",
    ) -> None:
        """
        Parameters
        ----------
        params :
            Input parameter for plotting. If `type` is 'lattice', it must be a 6-tuple
            containing the lengths a, b, c and angles alpha, beta, gamma. Otherwise, it
            must be a 3 by 3 numpy array with each row vector containing each
            real/reciprocal lattice vector. If not provided, a hexagonal lattice is
            shown by default.
        type : str, default: 'bvec'
            Specifies the type of the input parameters. Valid types are 'lattice',
            'avec', 'bvec'.
        """
        super().__init__()

        if params is None:
            params = np.array(
                [
                    [2.05238966, 0.0, 0.0],
                    [1.02619483, 1.77742159, 0.0],
                    [0.0, 0.0, 1.04510734],
                ]
            )
            type = "bvec"

        if type == "lattice":
            bvec = to_reciprocal(abc2avec(*params))
        elif type == "avec":
            bvec = to_reciprocal(params)
        elif type == "bvec":
            bvec = params

        self.controls = None
        self.plot = BZPlotWidget(bvec)
        self.setCentralWidget(self.plot)

        self.controls = LatticeWidget(bvec)
        self.controls.show()

        self.controls.sigChanged.connect(self.plot.set_bvec)


class LatticeWidget(QtWidgets.QTabWidget):
    sigChanged = QtCore.Signal(np.ndarray)

    def __init__(self, bvec):
        super().__init__()

        # self.setLayout(QtWidgets.QVBoxLayout(self))
        self.params_latt = ParameterGroup(
            ncols=3,
            **{
                "a": dict(qwtype="btspin", value=1, showlabel="𝑎"),
                "b": dict(qwtype="btspin", value=1, showlabel="𝑏"),
                "c": dict(qwtype="btspin", value=1, showlabel="𝑐"),
                "alpha": dict(
                    qwtype="btspin", value=90.0, minimum=0, maximum=180, showlabel="𝛼"
                ),
                "beta": dict(
                    qwtype="btspin", value=90.0, minimum=0, maximum=180, showlabel="𝛽"
                ),
                "gamma": dict(
                    qwtype="btspin", value=90.0, minimum=0, maximum=180, showlabel="𝛾"
                ),
                "apply": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Apply",
                    colspan="ncols",
                    clicked=self.latt_changed,
                ),
            },
        )
        self.params_avec = ParameterGroup(
            ncols=4,
            **{
                "_0": dict(widget=QtWidgets.QWidget(), showlabel=False, notrack=True),
                "_1": dict(
                    widget=QtWidgets.QLabel(
                        "𝑥", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "_2": dict(
                    widget=QtWidgets.QLabel(
                        "𝑦", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "_3": dict(
                    widget=QtWidgets.QLabel(
                        "𝑧", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "a1": dict(
                    widget=QtWidgets.QLabel("𝑎₁"), showlabel=False, notrack=True
                ),
                "a1x": dict(qwtype="btspin", value=1, showlabel=False),
                "a1y": dict(qwtype="btspin", value=1, showlabel=False),
                "a1z": dict(qwtype="btspin", value=1, showlabel=False),
                "a2": dict(
                    widget=QtWidgets.QLabel("𝑎₂"), showlabel=False, notrack=True
                ),
                "a2x": dict(qwtype="btspin", value=1, showlabel=False),
                "a2y": dict(qwtype="btspin", value=1, showlabel=False),
                "a2z": dict(qwtype="btspin", value=1, showlabel=False),
                "a3": dict(
                    widget=QtWidgets.QLabel("𝑎₃"), showlabel=False, notrack=True
                ),
                "a3x": dict(qwtype="btspin", value=1, showlabel=False),
                "a3y": dict(qwtype="btspin", value=1, showlabel=False),
                "a3z": dict(qwtype="btspin", value=1, showlabel=False),
                "apply": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Apply",
                    colspan="ncols",
                    clicked=self.avec_changed,
                ),
            },
        )
        for i in range(8):
            self.params_avec.layout().setColumnStretch(i, 1 if i < 2 else 6)
        self.params_bvec = ParameterGroup(
            ncols=4,
            **{
                "_0": dict(widget=QtWidgets.QWidget(), showlabel=False, notrack=True),
                "_1": dict(
                    widget=QtWidgets.QLabel(
                        "𝑥", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "_2": dict(
                    widget=QtWidgets.QLabel(
                        "𝑦", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "_3": dict(
                    widget=QtWidgets.QLabel(
                        "𝑧", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                    ),
                    showlabel=False,
                    notrack=True,
                ),
                "b1": dict(
                    widget=QtWidgets.QLabel("𝑏₁"), showlabel=False, notrack=True
                ),
                "b1x": dict(qwtype="btspin", value=1, showlabel=False),
                "b1y": dict(qwtype="btspin", value=1, showlabel=False),
                "b1z": dict(qwtype="btspin", value=1, showlabel=False),
                "b2": dict(
                    widget=QtWidgets.QLabel("𝑏₂"), showlabel=False, notrack=True
                ),
                "b2x": dict(qwtype="btspin", value=1, showlabel=False),
                "b2y": dict(qwtype="btspin", value=1, showlabel=False),
                "b2z": dict(qwtype="btspin", value=1, showlabel=False),
                "b3": dict(
                    widget=QtWidgets.QLabel("𝑏₃"), showlabel=False, notrack=True
                ),
                "b3x": dict(qwtype="btspin", value=1, showlabel=False),
                "b3y": dict(qwtype="btspin", value=1, showlabel=False),
                "b3z": dict(qwtype="btspin", value=1, showlabel=False),
                "apply": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Apply",
                    colspan="ncols",
                    clicked=self.bvec_changed,
                ),
            },
        )
        for i in range(8):
            self.params_bvec.layout().setColumnStretch(i, 1 if i < 2 else 6)
        self.addTab(self.params_latt, "𝑎𝑏𝑐𝛼𝛽𝛾")
        self.addTab(self.params_avec, "𝒂")
        self.addTab(self.params_bvec, "𝒃")

        self.set_bvec(bvec)
        self.bvec_changed()

        # self.params_latt.sigParameterChanged.connect(self.latt_changed)
        # self.params_avec.sigParameterChanged.connect(self.avec_changed)
        # self.params_bvec.sigParameterChanged.connect(self.bvec_changed)

    def block_params_signals(self, b: bool):
        self.params_latt.blockSignals(b)
        self.params_avec.blockSignals(b)
        self.params_bvec.blockSignals(b)

    def latt_changed(self):
        self.block_params_signals(True)
        self.set_avec(abc2avec(*self.latt_vals))
        self.block_params_signals(False)
        self.avec_changed()

    def avec_changed(self):
        self.block_params_signals(True)
        self.set_latt(*avec2abc(self.avec_val))
        self.block_params_signals(False)
        self.set_bvec(to_reciprocal(self.avec_val))
        self.bvec_changed()

    def bvec_changed(self):
        self.block_params_signals(True)
        self.set_avec(to_real(self.bvec_val))
        self.set_latt(*avec2abc(self.avec_val))
        self.block_params_signals(False)
        self.sigChanged.emit(self.bvec_val)

    def set_latt(self, a, b, c, alpha, beta, gamma):
        self.params_latt.set_values(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    def set_avec(self, avec):
        self.params_avec.set_values(
            **{f"a{i+1}{('x', 'y', 'z')[j]}": v for (i, j), v in np.ndenumerate(avec)}
        )

    def set_bvec(self, bvec):
        self.params_bvec.set_values(
            **{f"b{i+1}{('x', 'y', 'z')[j]}": v for (i, j), v in np.ndenumerate(bvec)}
        )

    @property
    def latt_vals(self):
        return tuple(
            self.params_latt.widgets[k].value()
            for k in ("a", "b", "c", "alpha", "beta", "gamma")
        )

    @property
    def avec_val(self):
        return np.array(
            [
                [self.params_avec.widgets[f"a{i}{c}"].value() for c in ("x", "y", "z")]
                for i in range(1, 4)
            ]
        )

    @property
    def bvec_val(self):
        return np.array(
            [
                [self.params_bvec.widgets[f"b{i}{c}"].value() for c in ("x", "y", "z")]
                for i in range(1, 4)
            ]
        )


class BZPlotWidget(QtWidgets.QWidget):
    def __init__(self, bvec):
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout(self))

        self.set_bvec(bvec, update=False)

        self._canvas = FigureCanvas(Figure())
        self.layout().addWidget(NavigationToolbar(self._canvas, self))
        self.layout().addWidget(self._canvas)

        self.ax = self._canvas.figure.add_subplot(projection="3d")
        self.ax.axis("off")

        self._lc = mpl_toolkits.mplot3d.art3d.Line3DCollection(
            self.lines, lw=1, linestyle="solid", clip_on=False
        )
        self.ax.add_collection(self._lc)

        # plot vertices
        self._plot = self.ax.plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2],
            ".",
            mew=0,
            clip_on=False,
        )[0]

        # plot reciprocal lattice vectors
        self._bvecs = []
        for i, b in enumerate(bvec):
            p = self.ax.plot(*[(0, bi) for bi in b], "-", c=f"C{i+1}", clip_on=False)
            t = self.ax.text(
                *(b + 0.15 * b / np.linalg.norm(b)),
                f"$b_{i+1}$",
                c=f"C{i+1}",
                ha="center",
                va="center_baseline",
            )
            self._bvecs.append((p[0], t))

        # plot origin
        self.ax.plot(0, 0, 0, ".", color="k", mew=0)

        # set camera
        self.ax.set_proj_type("persp", focal_length=np.inf)

        self.ax.view_init(elev=20, azim=-30, roll=0)

        # self._canvas.figure.tight_layout()

    def set_bvec(self, bvec, update=True):
        self.bvec = bvec
        self.lines, self.vertices = eplt.get_bz_edge(self.bvec, reciprocal=True)
        if update:
            self._update_canvas()

    def _update_canvas(self):
        for i, b in enumerate(self.bvec):
            self._bvecs[i][0].set_data_3d(*[(0, bi) for bi in b])
            self._bvecs[i][1].set_position_3d((b + 0.15 * b / np.linalg.norm(b)))

        self._lc.set_segments(self.lines)
        self._plot.set_data_3d(
            self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        )
        self._canvas.draw()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")
    app = BZPlotter()
    app.show()
    app.controls.activateWindow()
    app.controls.raise_()
    qapp.exec()