'''
New method for extracting the directional dark field with multi-resolution approach
'''

from .utils import *
import UMPA
import numpy as np
from matplotlib import pyplot as plt


from scipy.optimize import minimize_scalar, minimize
from scipy.optimize import golden
from datetime import datetime


#from cython_utils import abc_from_transform_c
from .utils import abc_from_transform_c_notc as abc_from_transform_c

class solver_at_resolution:
    def __init__(self, sams, refs, step, Nw, max_shift=5, initial_vals = None, max_sig = 10, blur_extra=0.45, pos_list = None, ROI=None):
        '''
        This model was not intended to be directly called by the user
        '''

        self.step = step
        self.Nw = Nw
        self.max_shift = max_shift
        self.max_sig = max_sig

        self.ROI = ROI

        self.PM = UMPA.model.UMPAModelDFKernel(sam_list=sams, ref_list=refs, pos_list=pos_list, mask_list=None, window_size=Nw, max_shift=max_shift, ROI=ROI)
        #self.PM.shift_mode = True
        self.PM.set_step(step)
        self.sh = self.PM.sh
        self.padding = self.PM.padding
        self.blur_extra = blur_extra

        # initial vals to be held in sx, sy, sig space
        if initial_vals is None:
            self.initial_vals = np.zeros(self.sh + (3,))
            self.initial_vals[:,:,2] = 0.4
        else:
            self.initial_vals = initial_vals

        self.final_vals = np.zeros_like(self.initial_vals)
        self.gauss_properties = np.zeros_like(self.initial_vals)

        self.result = None

        self.brent_vals = np.zeros_like(self.initial_vals)

    def initial_run_of_model(self):
        '''
        This step does the initial UMPA run to find translations due to refraction
        '''
        start = datetime.now()
        self.result = self.PM.match(self.step, abc_from_transform(self.initial_vals, self.blur_extra))
        self.result['dx'] = np.clip(self.result['dx'], -self.max_shift, self.max_shift)
        self.result['dy'] = np.clip(self.result['dy'], -self.max_shift, self.max_shift)
        print('Initial run to find phase image took: ' + str(datetime.now() - start))

    def cost_for_blur_new(self, v1, v2, v3, mode, pix_y, pix_x):
        '''
        Calculates the cost of blurring at pix_y pix_x using differeng gaussian parameters
        'mode' sets which order/tranform is used
        Note that co-ordinates are in OUTPUT frame of reference, not INPUT frame of reference, messing this us causes confusion!!!
        '''

        if mode == 'sig':
            sig = v1
            sx = v2
            sy = v3
        elif mode == 'mag':
            sx = v1 * np.cos(v2)
            sy = v1 * np.sin(v2)
            sig = v3
        elif mode == 'theta':
            sx = v2 * np.cos(v1)
            sy = v2 * np.sin(v1)
            sig = v3
        elif mode == 'together':
            sx = v1[0]
            sy = v1[1]
            sig = v1[2]
        else:
            print('mode not specified')
            raise NotImplementedError

        a, b, c, cost_eps = abc_from_transform_c(sx, sy, sig, self.blur_extra, self.max_sig)
        if self.ROI == None:
            umpa_pix_x = (pix_x * self.step) + self.padding
            umpa_pix_y = (pix_y * self.step) + self.padding
        else:
            umpa_pix_x = (pix_x * self.step) + self.padding + self.ROI[1][0]
            umpa_pix_y = (pix_y * self.step) + self.padding + self.ROI[0][0]

        cost = self.PM.cost(umpa_pix_y, umpa_pix_x, self.result['dy'][pix_y, pix_x], self.result['dx'][pix_y, pix_x], a, b, c)[0]

        return cost + cost_eps

    def optimise_pixel(self, pix_y, pix_x, method='golden', maxiter=20, tol=None, mode='coordinate_descent'):
        '''
        Optimises a pixel
        '''
        method = 'golden'

        sx = self.initial_vals[pix_y, pix_x, 0]
        sy = self.initial_vals[pix_y, pix_x, 1]
        sig = self.initial_vals[pix_y, pix_x, 2]


        if mode == 'coordinate_descent':
            for i in range(1):

                mag = np.sqrt(sx ** 2 + sy ** 2)
                # for first step need to set mag to be non-zero to find theta
                if mag == 0:
                    mag = 0.5

                sig = minimize_scalar(self.cost_for_blur_new, bounds=(0.1, 4), args=(sx, sy, 'sig', pix_y, pix_x), method=method, tol=tol, options={'maxiter': maxiter})['x']

                theta = minimize_scalar(self.cost_for_blur_new, bounds=(- np.pi/2, np.pi/2), args=(mag, sig, 'theta', pix_y, pix_x), tol = 1e-12, method=method, options={'maxiter': maxiter})['x']

                mag = minimize_scalar(self.cost_for_blur_new, bounds=(0, 1), args=(theta, sig, 'mag', pix_y, pix_x), tol = 1e-12, method=method, options={'maxiter': maxiter})['x']

                sx = mag * np.cos(theta)
                sy = mag * np.sin(theta)
        else:
            res = minimize(self.cost_for_blur_new, x0=[sx, sy, sig], args = ('fake', 'argument', 'together', pix_y, pix_x), method='Nelder-Mead', options={'maxiter': maxiter, 'fatol': 1e-12})
            sx = res['x'][0]
            sy = res['x'][1]
            sig = res['x'][2]

        self.final_vals[pix_y, pix_x, :] = [sx, sy, sig]
        return [sx, sy, sig]

    def optimise_image(self, method='golden', maxiter=20, tol=None, mode = 'coordinate_descent', transmission_threshold = 5.99):
        '''
        Optimises the whole image
        '''
        start = datetime.now()
        # NEED TO FIGURE OUT WHY CORE DUMPS FOR (0,0) - sholdnt need range(1,sh) here
        print('optimising with {} mode'.format(mode))
        for i in range(self.sh[0]):
            for j in range(self.sh[1]):
                if self.result['T'][i,j] < transmission_threshold:
                    self.optimise_pixel(i, j, method=method, maxiter=maxiter, tol=tol, mode = mode)


            #print('done ' + str((i * self.sh[1]) + j) + ' pixels of ' + str((self.sh[0]) * self.sh[1]))

        print(str((self.sh[0] - 1) * self.sh[1]) + ' pixels took ' + str(datetime.now() - start))
        print('Average pixel optimisation time was: ' + str((datetime.now() - start)/((self.sh[0] - 1) * self.sh[1])))

    def return_rgb(self, sigma_max=True, log_scale=False):
        '''
        Return an RGB image
        '''

        # cant see an easier way to do this
        if sigma_max==True:
            sigma_max = self.max_sig
            print('using defaul maximum sigma value for rendering RGB image')

        self.gauss_properties = gauss_from_transform(self.final_vals, sigma_max)
        rgb, sig_same, sig_diff, direction = generate_rgb(self.gauss_properties, log_scale=log_scale)

        rgb = np.clip(rgb, 0, 255) # some illegal values manages to sneak through
        return rgb


if __name__ == "__main__":
    sim = UMPA.utils.prep_simul()

    sams = sim['meas']
    refs = sim['ref']
    sams = sams / np.max(sams[0])
    refs = refs / np.max(sams[0])

    model = solver_at_resolution(sams, refs, step=40, Nw=5, blur_extra = 0.05)
    model.initial_run_of_model()
    model.optimise_image()
    rgb = model.return_rgb(sigma_max = 2, log_scale= False)

    plt.figure()
    plt.imshow(rgb)
    plt.show()


