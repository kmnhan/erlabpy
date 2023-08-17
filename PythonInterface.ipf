#pragma rtGlobals=1		// Use modern global access method.

Function LoadPythonArray(outName, [pathName, fileName])
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
	
	// Rename the loaded wave.
	Duplicate /O '__xarray_dataarray_variable__', $outName; KillWaves '__xarray_dataarray_variable__'
	
	Print "Loaded Python DataArray " + outName + " from " + S_path + S_fileName
	
End

