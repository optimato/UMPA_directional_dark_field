# Importing the key pieces
import numpy as np
from matplotlib import pyplot as plt
import h5py
import UMPA
import UMPA_directional_dark_field as uddf

# Firstly we need to prepare the data
# Here you would load your own data but we will simulate a speckle experiment

sim = UMPA.utils.prep_simul()

sams = sim['meas']
refs = sim['ref']

# here are some arguments UMPA takes
final_nw = 5 #  float: the size of the analysis window will be 2*nw+1
final_step = 5 # float: used to speed up test reconstructions. Model will be run every n pixels (ie if 5, will only run every 5th pixel in each direction)

# the scrip automatically saves the results - put your own path here
save_path = ''
save_path = save_path + 'test_reconstruction'
savename = save_path + '_step_' + str(final_step) + '_Nw_' + str(final_nw)

# There are 3 implementations of the method here, all take the same arguments and produce the same result with some
# differences in the multiprocessing approach
# They will create a final .h5 file which contains the result and a .png to quickly check it's ok
# The multiprocessing variations will also save some temp files which can be delected once it has run

# This does not use multiprocessing at all
#rgb = uddf.do_it_all_for_me(sams, refs, savename, final_nw=final_nw, final_step=final_step, pos_list=None,
#                                       sigma_max=1.5, blur_extra=0.05)

# This is a simple multiprocessing version, however it is not the most memory efficient
# This uses the python multiprocessing module and may not work properly with windows
rgb = uddf.do_it_all_for_me_multiprocessing(sams, refs, savename, final_nw=final_nw, final_step=final_step, pos_list=None,
                                       sigma_max=1.5, blur_extra=0.05, n_process=8)

# This is a more complex multiprocessing version, it is more memory efficient but isnt compatable with the 'sample
# stepping' method described in our other new paper
# This uses the python multiprocessing module and may not work properly with windows
#rgb = uddf.do_it_all_for_me_multiprocessing_memsave(sams, refs, savename, final_nw=final_nw, final_step=final_step, pos_list=None,
#                                       sigma_max=1.5, blur_extra=0.05, n_process=8)

# We can look at the output and look at the other images

F = h5py.File(savename + '.h5', 'r')
gp = np.array(F['gauss_properties_stage_2'])
dpcx = np.array(F['dpcx_stage_2'])
dpcy = np.array(F['dpcy_stage_2'])
t = np.array(F['transmission_image_stage_2'])
F.close()

# As an example, we can regenerate the RGB image after clipping the gaussian kernel array
# Here we are clipping both standard deviations to 3 as an example

gp2 = gp.copy()
gp2[:,:,0] = np.clip(gp2[:,:,0], 0 ,3)
gp2[:,:,1] = np.clip(gp2[:,:,1], 0 ,3)

rgb2, magnitude, directionality, orientation = uddf.generate_rgb(gp2)

plt.figure()
plt.imshow(rgb)
plt.show()