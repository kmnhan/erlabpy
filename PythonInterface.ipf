#pragma rtGlobals=1		// Use modern global access method.

// Load HDF5 files (.h5) exported from ERLabPy.
// Only works with `xarray.DataArray`s exported using `erlab.io.save_as_hdf5` with `igor_compat=True`.
// Currently only loads the values and coordinates (attributes are stripped).

// Version 1.00, August 17, 2023: Initial release.

#pragma version = 1.0

#pragma ModuleName = PythonInterface

Menu "Load Waves"
	"Load ERLabPy HDF5...", /Q, LoadPythonArray()
End

Function LoadPythonArray([outName, pathName, fileName])
	// Load .h5 files saved from python

	String outName	 // Name of loaded wave
	String pathName	 // Name of symbolic path
	String fileName	 // Name of HDF5 file

	if( ParamIsDefault(pathName) )
		pathName = ""
	endif

	if( ParamIsDefault(fileName) )
		fileName = ""
	endif

	Variable fileID	// HDF5 file ID will be stored here

	// Open the HDF5 file.
	HDF5OpenFile /P=$pathName /R fileID as fileName

	// Load the HDF5 dataset.
	HDF5LoadData /O /Q /IGOR=-1 fileID, "__xarray_dataarray_variable__"

	// Close the HDF5 file.
	HDF5CloseFile fileID

	if( ParamIsDefault(outName) )
		outName = RemoveEnding(S_fileName, ".h5")
		Prompt outName, "Name: "
		DoPrompt "Enter Wave Name", outName
		if (V_Flag)
      	return -1
		endif
	endif

	// Rename the loaded wave.
	Duplicate /O '__xarray_dataarray_variable__', $outName; KillWaves '__xarray_dataarray_variable__'

	Print "Loaded Python DataArray as " + outName + " from " + S_path + S_fileName

End
