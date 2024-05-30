import sys
from typing import Literal, cast

import mpl_toolkits
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets

import erlab.plotting.erplot as eplt
from erlab.interactive.utils import ParameterGroup
from erlab.lattice import abc2avec, avec2abc, to_real, to_reciprocal


class BZPlotter(QtWidgets.QMainWindow):
    """
    Interactive Brillouin zone plotter.

    Parameters
    ----------
    params :
        Input parameter for plotting. If `param_type` is 'lattice', it must be a
        6-tuple containing the lengths a, b, c and angles alpha, beta, gamma.
        Otherwise, it must be a 3 by 3 numpy array with each row vector containing
        each real/reciprocal lattice vector. If not provided, a hexagonal lattice is
        shown by default.
    param_type
        Specifies the param_type of the input parameters. Valid param_types are
        `'lattice'`, `'avec'`, `'bvec'`. By default, `'bvec'` is assumed.
    execute
        If `True`, the Qapp instance will be executed immediately.
    """

    def __init__(
        self,
        params: tuple[float, ...] | npt.NDArray[np.float64] | None = None,
        param_type: Literal["lattice", "avec", "bvec"] | None = None,
        execute: bool = True,
    ) -> None:
        self.qapp = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)

        super().__init__()

        if params is None:
            params = np.array(
                [
                    [2.05238966, 0.0, 0.0],
                    [1.02619483, 1.77742159, 0.0],
                    [0.0, 0.0, 1.04510734],
                ]
            )
            param_type = "bvec"

        if param_type == "lattice":
            if len(params) != 6:
                raise TypeError("Lattice parameters must be a 6-tuple.")

            bvec = to_reciprocal(abc2avec(*params))
        else:
            if not isinstance(params, np.ndarray):
                raise TypeError("Lattice vectors must be a numpy array.")
            if params.shape != (3, 3):
                raise TypeError("Lattice vectors must be a 3 by 3 numpy array.")

            if param_type == "avec":
                bvec = to_reciprocal(params)
            elif param_type == "bvec":
                bvec = params

        self.plot = BZPlotWidget(bvec)
        self.setCentralWidget(self.plot)

        self.controls = LatticeWidget(bvec)
        self.controls.sigChanged.connect(self.plot.set_bvec)

        self.__post_init__(execute=execute)

    def __post_init__(self, execute=None):
        self.show()
        self.activateWindow()
        self.raise_()
        self.controls.show()

        if execute is None:
            execute = True
            try:
                shell = get_ipython().__class__.__name__  # type: ignore
                if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                    execute = False
            except NameError:
                pass
        if execute:
            self.qapp.exec()


class LatticeWidget(QtWidgets.QTabWidget):
    sigChanged = QtCore.Signal(np.ndarray)  #: :meta private:

    def __init__(self, bvec):
        super().__init__()

        # self.setLayout(QtWidgets.QVBoxLayout(self))
        self.params_latt = ParameterGroup(
            ncols=3,
            a={"qwtype": "btspin", "value": 1, "showlabel": "𝑎", "decimals": 5},
            b={"qwtype": "btspin", "value": 1, "showlabel": "𝑏", "decimals": 5},
            c={"qwtype": "btspin", "value": 1, "showlabel": "𝑐", "decimals": 5},
            alpha={
                "qwtype": "btspin",
                "value": 90.0,
                "minimum": 0,
                "maximum": 180,
                "showlabel": "𝛼",
                "decimals": 5,
            },
            beta={
                "qwtype": "btspin",
                "value": 90.0,
                "minimum": 0,
                "maximum": 180,
                "showlabel": "𝛽",
                "decimals": 5,
            },
            gamma={
                "qwtype": "btspin",
                "value": 90.0,
                "minimum": 0,
                "maximum": 180,
                "showlabel": "𝛾",
                "decimals": 5,
            },
            apply={
                "qwtype": "pushbtn",
                "notrack": True,
                "showlabel": False,
                "text": "Apply",
                "colspan": "ncols",
                "clicked": self.latt_changed,
            },
        )
        self.params_avec = ParameterGroup(
            ncols=4,
            _0={
                "widget": QtWidgets.QWidget(),
                "showlabel": False,
                "notrack": True,
            },
            _1={
                "widget": QtWidgets.QLabel(
                    "𝑥", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            _2={
                "widget": QtWidgets.QLabel(
                    "𝑦", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            _3={
                "widget": QtWidgets.QLabel(
                    "𝑧", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            a1={
                "widget": QtWidgets.QLabel("𝑎₁"),
                "showlabel": False,
                "notrack": True,
            },
            a1x={"qwtype": "btspin", "value": 1, "showlabel": False},
            a1y={"qwtype": "btspin", "value": 1, "showlabel": False},
            a1z={"qwtype": "btspin", "value": 1, "showlabel": False},
            a2={
                "widget": QtWidgets.QLabel("𝑎₂"),
                "showlabel": False,
                "notrack": True,
            },
            a2x={"qwtype": "btspin", "value": 1, "showlabel": False},
            a2y={"qwtype": "btspin", "value": 1, "showlabel": False},
            a2z={"qwtype": "btspin", "value": 1, "showlabel": False},
            a3={
                "widget": QtWidgets.QLabel("𝑎₃"),
                "showlabel": False,
                "notrack": True,
            },
            a3x={"qwtype": "btspin", "value": 1, "showlabel": False},
            a3y={"qwtype": "btspin", "value": 1, "showlabel": False},
            a3z={"qwtype": "btspin", "value": 1, "showlabel": False},
            apply={
                "qwtype": "pushbtn",
                "notrack": True,
                "showlabel": False,
                "text": "Apply",
                "colspan": "ncols",
                "clicked": self.avec_changed,
            },
        )
        for i in range(8):
            self.params_avec.layout().setColumnStretch(i, 1 if i < 2 else 6)
        self.params_bvec = ParameterGroup(
            ncols=4,
            _0={
                "widget": QtWidgets.QWidget(),
                "showlabel": False,
                "notrack": True,
            },
            _1={
                "widget": QtWidgets.QLabel(
                    "𝑥", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            _2={
                "widget": QtWidgets.QLabel(
                    "𝑦", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            _3={
                "widget": QtWidgets.QLabel(
                    "𝑧", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
                ),
                "showlabel": False,
                "notrack": True,
            },
            b1={
                "widget": QtWidgets.QLabel("𝑏₁"),
                "showlabel": False,
                "notrack": True,
            },
            b1x={"qwtype": "btspin", "value": 1, "showlabel": False},
            b1y={"qwtype": "btspin", "value": 1, "showlabel": False},
            b1z={"qwtype": "btspin", "value": 1, "showlabel": False},
            b2={
                "widget": QtWidgets.QLabel("𝑏₂"),
                "showlabel": False,
                "notrack": True,
            },
            b2x={"qwtype": "btspin", "value": 1, "showlabel": False},
            b2y={"qwtype": "btspin", "value": 1, "showlabel": False},
            b2z={"qwtype": "btspin", "value": 1, "showlabel": False},
            b3={
                "widget": QtWidgets.QLabel("𝑏₃"),
                "showlabel": False,
                "notrack": True,
            },
            b3x={"qwtype": "btspin", "value": 1, "showlabel": False},
            b3y={"qwtype": "btspin", "value": 1, "showlabel": False},
            b3z={"qwtype": "btspin", "value": 1, "showlabel": False},
            apply={
                "qwtype": "pushbtn",
                "notrack": True,
                "showlabel": False,
                "text": "Apply",
                "colspan": "ncols",
                "clicked": self.bvec_changed,
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
            **{f"a{i + 1}{('x', 'y', 'z')[j]}": v for (i, j), v in np.ndenumerate(avec)}
        )

    def set_bvec(self, bvec):
        self.params_bvec.set_values(
            **{f"b{i + 1}{('x', 'y', 'z')[j]}": v for (i, j), v in np.ndenumerate(bvec)}
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
            p = self.ax.plot(*[(0, bi) for bi in b], "-", c=f"C{i + 1}", clip_on=False)
            t = self.ax.text(
                *(b + 0.15 * b / np.linalg.norm(b)),
                f"$b_{i + 1}$",
                c=f"C{i + 1}",
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
            self._bvecs[i][1].set_position_3d(b + 0.15 * b / np.linalg.norm(b))

        self._lc.set_segments(self.lines)
        self._plot.set_data_3d(
            self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        )
        self._canvas.draw()
