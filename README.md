# UMPA_directional_dark_field

A Directional Dark-Field expansion to the UMPA algorithm. 
As Described in ['X-ray directional dark-field imaging using Unified Modulated Pattern Analysis' Paper by Ronan Smith et.al (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273315) - please cite this paper if you use our code.

# Installation Instructions 

[This code relies on the C++ version of UMPA, and has the same dependencies. It can be found here, please follow thier installation instructions.](https://github.com/optimato/UMPA) 

After installing the c++ version of UMPA, the simplest method of instaling this package is navigating to the folder containing the setup.py file and installing with: 

```
pip install . --user
```

There is an example script in this folder showing how to run a simple reconstruction. 

# System Requirements

This has been tested on various Linux distrubutions and should work on Windows, however this is not guaranteed. The main UMPA Code will not currently work on MacOS but if this changes, this code should be compatible.

This needs Python 3.10 or above to use multiprocessing, but will run on older versions if this is not used. 

Sometimes reconstructions hang when launched from a Jupyter notebook, if you encounter problems please try standard python.
