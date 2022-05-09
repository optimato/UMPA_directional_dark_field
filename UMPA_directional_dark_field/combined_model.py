from .utils import *
import UMPA
import numpy as np
from matplotlib import pyplot as plt
import h5py

from datetime import datetime

from .single_model import solver_at_resolution
from scipy import ndimage

class multi_resolution_solver:
    def __init__(self, sams, refs, final_step, final_nw, n_iters_bin=3, n_iters_final=2, max_shift=5, step_multiplier=3,
                 max_sig=10, blur_extra=0.45, pos_list=None, ROI=None):
        '''
        sams -> list of sample images
        refs -> list of referecce images
        final_step -> int, the requird step size of the final reconstrcution
        final_nw -> int, the final UMPA window will have size (2*nw +1)
        max_shift -> int, maximum phase shift passed to UMPA
        step multiplier -> int, the undersampling to be applied between iterations
        max_sig -> float, the maximum sigma value of the directional dark-field gaussian kernel
        blur_extra -> float, the sigma of the antialiasing gaussian kernel
        pos_list -> list of positions if using sample stepping (See UMPA)
        ROI -> region of interest for rectonstruction (See UMPA)

        '''

        self.single_res_models = []
        self.n_iters_total = n_iters_bin + n_iters_final
        self.max_sig = max_sig
        self.n_iters_bin = n_iters_bin
        self.n_iters_final = n_iters_final

        self.stage = 0
        self.blur_extra = blur_extra

        for i in range(n_iters_bin):
            self.single_res_models.append(solver_at_resolution(sams, refs, step=final_step*(step_multiplier**(n_iters_bin - i - 1)), Nw=final_nw, max_shift=max_shift, max_sig= self.max_sig, blur_extra = self.blur_extra, pos_list = pos_list, ROI=ROI))
            print('Created model number {}, step is {}'.format(i, final_step * (step_multiplier ** (n_iters_bin - i - 1))))
        for i in range(n_iters_final):
            self.single_res_models.append(solver_at_resolution(sams, refs, step=final_step, Nw=final_nw, max_shift=max_shift, max_sig=self.max_sig, blur_extra = self.blur_extra, pos_list=pos_list, ROI=ROI))
            print('Created final {}, step is {}'.format(i, final_step))


    def solve_model(self, stage = None, method='golden', maxiter=20, tol=None, mode = None):
        '''
        Optimises every pixel in the image at a given stage
        stage -> int, stage to solve at (None for default)
        method -> string, the solver to use - see scipy.optimise for details, default is 'golden'
        maxiter -> int, the maximum number of iterations of the solver at each pixel
        tol -> float, the tolerance of the solver (None to use default)
        mode -> string. 'coordinate_descent' or 'gradient_descent'. None for script to choose best option.
        '''
        if stage is None:
            stage = self.stage

        if mode is None:
            if stage <= 1:
                mode = 'coordinate_descent'
            else:
                mode = 'gradient_descent'

        self.single_res_models[stage].initial_run_of_model()
        self.single_res_models[stage].optimise_image(method=method, maxiter=maxiter, tol=tol, mode = mode)
        rgb = self.single_res_models[stage].return_rgb()
        return rgb

    def send_output_to_next_model(self, stage=None, blur=None):
        '''
        Sends the output of one iteration to become the input of the next after applying upscaling and blurring the image
        to prevent aliasing
        stage -> int, the stage you are passing from. Use None for default.
        blur -> int, the size of the kernel for bluring. Use None for default.

        '''

        if stage is None:
            stage = self.stage
            self.stage += 1
            if stage == 0 or stage ==1:
                blur = 9
            else:
                blur = 3

        #removing the bits that have the opposite symmetry to make bluring work

        output_without_symmetry = np.zeros_like(self.single_res_models[stage].final_vals)

        for i in range(output_without_symmetry.shape[0]):
            for j in range(output_without_symmetry.shape[1]):
                if self.single_res_models[stage].final_vals[i, j, 0] < 0:
                    output_without_symmetry[i, j, 0] = -1 * self.single_res_models[stage].final_vals[i, j, 0]
                    output_without_symmetry[i, j, 1] = -1 * self.single_res_models[stage].final_vals[i, j, 1]
                else:
                    output_without_symmetry[i, j, 0] = self.single_res_models[stage].final_vals[i, j, 0]
                    output_without_symmetry[i, j, 1] = self.single_res_models[stage].final_vals[i, j, 1]

                output_without_symmetry[i, j, 2] = abs(self.single_res_models[stage].final_vals[i, j, 2])

        if stage < self.n_iters_total - 1:

            #output_without_symmetry = self.single_res_models[stage].final_vals
            #upscale
            new_arr = np.zeros_like(self.single_res_models[stage+1].initial_vals)

            scale_y = self.single_res_models[stage].final_vals.shape[0] / self.single_res_models[stage+1].initial_vals.shape[0]
            scale_x = self.single_res_models[stage].final_vals.shape[1] / self.single_res_models[stage+1].initial_vals.shape[1]

            for i in range(new_arr.shape[0]):
                for j in range(new_arr.shape[1]):
                    py = int((i * scale_y) // 1)
                    px = int((j * scale_x) // 1)
                    new_arr[i, j, :] = output_without_symmetry[py, px, :]

            # blur
            new_arr[:, :, 0] = ndimage.percentile_filter((new_arr[:,:,0]), 50, size=(blur, blur), mode='reflect') #* np.sign(ndimage.median_filter(new_arr[:,:,0], size=(blur, blur), mode='reflect'))
            new_arr[:, :, 1] = ndimage.percentile_filter((new_arr[:,:,1]), 50, size=(blur, blur), mode='reflect') #* np.sign(ndimage.median_filter(new_arr[:,:,1], size=(blur, blur), mode='reflect'))
            new_arr[:, :, 2] = ndimage.median_filter((new_arr[:,:,2]), size=(blur, blur), mode='reflect')

            self.single_res_models[stage+1].initial_vals = new_arr
            print(f'successfully passed to next model and blurred with kernel size {blur}')

        else:
            print('Stage out of range')


def do_it_all_for_me(sams, refs, save_path=None, final_nw=5, final_step=5, pos_list=None, sigma_max=15, max_shift=5, ROI = None, blur_extra=0.45, savepng=True):
    '''
    A function for running everything
    sams -> list of sample frames
    refs -> list of reference frames
    savepath -> string containing path to save, None for no saving
    final_nw -> int, (2*final_nw +1) is window size in final iteration
    final_step -> int, the UMPA step size in the final iteration
    post_list -> list of positions if using sample stepping, None for diffuser stepping
    sigma_max -> float, the maximum allowable sigma for the gaussian blur
    max_shift -> float, the maximum displacement for phase shift in UMPA
    ROI -> Region of interest where image will be analysed (See UMPA), None to use whole image
    blur_extra -> float, the kernel size for anti-alisaising kernel
    '''

    num_frames = len(sams)
    max_iter_final = 500
    if save_path is not None:
        savename = save_path + '_N_' + str(num_frames) + '_step_' + str(final_step) + '_Nw_' + str(
            final_nw) + '_final_max_iter_' + str(max_iter_final)

    big_model = multi_resolution_solver(sams, refs, final_step, final_nw, step_multiplier=3, n_iters_final=0, max_shift=max_shift, pos_list=pos_list, ROI=ROI, blur_extra=blur_extra)

    for i in range(big_model.n_iters_total):
        if i < big_model.n_iters_bin - 1:
            rgb = big_model.solve_model(maxiter=100, tol=1e-15, mode='coordinate_descent')
        else:
            rgb = big_model.solve_model(maxiter=max_iter_final, tol=1e-15, mode='gradient_descent')

        rgb = big_model.single_res_models[i].return_rgb(sigma_max=sigma_max, log_scale=True)

        if save_path is not None:

            if savepng:
                fig = plt.figure(figsize=(7, 4))
                plt.imshow(rgb)
                plt.title('Reconstrution stage ' + str(i + 1))
                plt.savefig(savename + 'recon_stage_' + str(i) + '.png')
                plt.close(fig)

            F = h5py.File(savename + '.h5', 'a')
            F.create_dataset('gauss_properties_stage_' + str(i), data=big_model.single_res_models[i].gauss_properties)
            F.create_dataset('final_vals_stage_' + str(i), data=big_model.single_res_models[i].final_vals)
            F.create_dataset('transmission_image_stage_' + str(i), data=big_model.single_res_models[i].result['T'])
            F.create_dataset('dpcx_stage_' + str(i), data=big_model.single_res_models[i].result['dx'])
            F.create_dataset('dpcy_stage_' + str(i), data=big_model.single_res_models[i].result['dy'])
            F.create_dataset('initial_vals_' + str(i), data=big_model.single_res_models[i].initial_vals)
            F.close()

        big_model.send_output_to_next_model()

    print('done')
    return big_model


def test_reconstruction(save_data=False):
    '''
    Test Code
    '''

    final_nw = 5
    final_step =10
    num_frames = 25
    max_iter_final = 200

    save_path = 'test_reconstruction'
    savename = save_path + '_N_' + str(num_frames) + '_step_' + str(final_step) + '_Nw_' + str(
        final_nw) + '_final_max_iter_' + str(max_iter_final)

    sim = UMPA.utils.prep_simul()

    sams = sim['meas']
    refs = sim['ref']

    if save_data == False:
        big_model = do_it_all_for_me(sams, refs, savepath=None, final_nw=final_nw, final_step=final_step, pos_list=None, sigma_max=1.5, blur_extra=0.05)
    else:
        big_model = do_it_all_for_me(sams, refs, savepath=savename, final_nw=final_nw, final_step=final_step, pos_list=None, sigma_max=1.5, blur_extra=0.05)
    return big_model

if __name__ == "__main__":
    import h5py
    big_model = test_reconstruction(save_data=False)
    print('Its done')

