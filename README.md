# UMPA_directional_dark_field

Directional Dark Field addition to UMPA
As Desribed in 'X-ray directional dark-field imaging using Unified Modulated Pattern Analysis' Paper by Ronan Smith et.al (2022) - please cite this paper if you use our code

[This code relies on the C++ version of UMPA, which can be found here](https://github.com/optimato/UMPA)

The simplest method of installation is navigating to the folder containing the setup.py file and install with 

'''
pip install . --user'
'''

There is an example script showing how to run a simple reconstruction

# System Requirements

This has been tested on various Linux distrubutions but should work on Windows. The main UMPA Code will not currently work on Apple but once this is fixed, this code should be compatible.

This needs Python 3.10 or above to use multiprocessing, but will run on older versions if this is not used. 

Sometimes reconstructions hang when launched from a Jupyter notebook, if you encounter problems please try standard python.
