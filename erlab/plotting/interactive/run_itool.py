from erlab.plotting.interactive.imagetool_new import itool_
import pyqtgraph as pg
# pg.setConfigOption('useNumba', True)
import xarray as xr

# data = erlab.io.load_igor_pxp(
#     "/Users/khan/Documents/ERLab/TiSe2/220221_KRISS/220221_KRISS_TiSe2.pxp",
#     silent=True,
# )["f043"]
data = xr.open_dataarray(
    # "/Users/khan/Documents/ERLab/TiSe2/kxy10.nc"
    "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
    # "/Users/khan/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d.nc"
)
itool_(data)



# if __name__ == "__main__":
#     qapp = QtWidgets.QApplication.instance()
#     if not qapp:
#         qapp = QtWidgets.QApplication(sys.argv)
#     qapp.setStyle("Fusion")
#     data = xr.open_dataarray(
#         # "/Users/khan/Documents/ERLab/TiSe2/kxy10.nc"
#         "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
#         # "/Users/khan/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d.nc"
#     )
#     # import erlab.io

#     # data = erlab.io.load_igor_pxp(
#     #     "/Users/khan/Documents/ERLab/TiSe2/220221_KRISS/220221_KRISS_TiSe2.pxp",
#     #     silent=True,
#     # )["f043"]
#     # demo = ImageSlicerArea()
#     # demo.set_data(data)
#     demo = ImageTool(data=data)
#     demo.show()
#     demo.raise_()
#     qapp.exec()
#     # demo.profiles[0].refresh_data(0,None)
#     # demo.mainimage.refresh_image()

#     # sys.exit(app.exec_())