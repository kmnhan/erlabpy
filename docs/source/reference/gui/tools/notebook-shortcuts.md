(interactive-misc-magics)=

# Notebook shortcuts

Loading the {mod}`erlab.interactive` IPython extension (`%load_ext erlab.interactive`) registers convenient line magics for quickly launching the various interactive tools from within a Jupyter notebook or IPython console.

```ipython
%ktool --cmap viridis darr.sel(eV=0)

%dtool darr

%goldtool darr.isel(beta=1)

%restool darr.mean(dim='kx')

%ftool --model my_model darr
```
