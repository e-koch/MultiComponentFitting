# Example run for minicube_fit with three components

import numpy as np
import pylab as pl
import time
from scipy import stats
from scipy.optimize import minimize, basinhopping, least_squares
from scipy.signal import medfilt
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

# from pymultinest.solve import solve
# from pymultinest import Analyzer
# import os

# from gausspy.tvdiff import TVdiff_adapt, TVdiff

from multicomponentfitting.spatially_aware_fitting.minicube_fit import \
    (minicube_model, minicube_gaussmodel)
from multicomponentfitting.spatially_aware_fitting.cube_reg_model import \
    (spatial_reg_cube_model, gaussian, spatial_reg_cube_model_onoff, generate_bounds)
from multicomponentfitting.spatially_aware_fitting.cube_reg_model_fast import \
    (spatial_reg_cube_model_cyth, spatial_reg_cube_model_conv_cyth,
     gaussian_jacobian, spatial_reg_jacobian)



from multicomponentfitting.spatially_aware_fitting.component_guesser import \
    (triple_point_find, reg_derivs)
from multicomponentfitting.spatially_aware_fitting.tvdiff_spatial import \
    (TVdiff_adapt_cube)
from multicomponentfitting.nonparametric.whittaker_smoother import \
    (whittaker_smooth, whittaker_smooth_auto, whittaker_smooth_morozov_iter,
     whittaker_smooth_gcv, whittaker_smooth_mscr)

# Set the seed for now
np.random.seed(45389348)

num_pts = 150
npix = 10
ncomps = 3

noise = 0.25

amp = 5.
mean = num_pts / 2.
# std = 5.
std = 3.

# neigb_scale = npix / 2.
# neigb_scale = 3.
neigb_scale = 3.

source_width = 3.

vels = np.arange(0, num_pts, 1)

# model1, params1 = minicube_model(vels,
#                                  amp, -0.6, 0.2,
#                                  mean, 1, -2,
#                                  std, -0.1, 0.1, npix=npix)

model1, params1 = minicube_gaussmodel(vels,
                                      amp, source_width, (npix/3, npix/3),
                                      mean, 1, -2,
                                      # std, -0.1, 0.1, npix=npix)
                                      std, 0.0, 0.0, npix=npix)

amp2 = 3.
mean2 = num_pts / 2.
# std2 = 5.
std2 = 3.

# model2, params2 = minicube_model(vels,
#                                  amp2, 0.35, -0.35,
#                                  mean2, -1, 2,
#                                  std2, 0.1, -0.1, npix=npix)

model2, params2 = minicube_gaussmodel(vels,
                                      amp2, source_width, (npix/2, npix/2),
                                      mean2, -1, 2,
                                      std2, 0.0, -0.0, npix=npix)
amp3 = 3.
mean3 = 8 * num_pts / 10.
# std3 = 5.
std3 = 3.

# amp3 = 3.
# mean3 = num_pts / 2.
# std3 = 15.

model3, params3 = minicube_gaussmodel(vels,
                                      amp3, source_width, (npix * 0.75, npix * 0.25),
                                      mean3, -0.1, 0.1,
                                      std3, 0.0, 0.0, npix=npix)

# model3, params3 = minicube_gaussmodel(vels,
#                                       amp3, 2., (1, 1),
#                                       mean3, 1, -2,
#                                       std3, 0.0, 0.0, npix=npix)

# simulate a "real sky" - no negative emission.
model1[model1 < 0] = 0
model2[model2 < 0] = 0
model3[model3 < 0] = 0

params1[:, :, 0][params1[:, :, 0] < 0] = 0.
params2[:, :, 0][params2[:, :, 0] < 0] = 0.
params3[:, :, 0][params3[:, :, 0] < 0] = 0.

model_errs = noise * np.ones_like(model1[0])  # * u.K

model = model1 + model2 + model3
# model = model1 + model2

noisy_model = model + np.random.normal(0, noise, model.shape)
# Add the 'on' parameter
# ons = (params1[..., 0] > 0).astype(float)
# ons[params1[..., 0] == 0.] = -1.
# params1 = np.dstack([ons, params1])

# ons2 = (params2[..., 0] > 0).astype(float)
# ons2[params2[..., 0] == 0.] = -1.
# params2 = np.dstack([ons2, params2])

params = np.dstack([params1, params2, params3])
# params = np.dstack([params1, params2])
# params = params1

# Try guessing the components and parameters
yy, xx = np.indices(model.shape[1:])

# # a1 = 4.
# a1 = 0.01

# cumul_smooth = 21
# spec_smooth = 11

# sqrt_2 = np.sqrt(2)

# import matplotlib.pyplot as plt

# lambdas = np.zeros_like(model[0])

# for y, x in zip(yy.ravel(), xx.ravel()):
# # for y, x in zip(yy.ravel()[::-1], xx.ravel()[::-1]):
# # for y, x in zip([3], [3]):

#     spectral_data = noisy_model[:, y, x]
#     # spectral_data = model[:, y, x]

#     # ADD SMOOTHING STEPS
#     # smooth_data = np.gradient(medfilt(np.cumsum(spectral_data), cumul_smooth))

#     # Remaining negative regions likely from remaining bowls
#     # smooth_data[smooth_data < 0.] = 0.

#     # smooth_data = medfilt(smooth_data, spec_smooth)
#     # smooth_data = whittaker_smooth(spectral_data, 0.01832, d=3)
#     # smooth_data, opt_out = whittaker_smooth_auto(spectral_data, 100., d=2)
#     # smooth_data, opt_out = \
#     #     whittaker_smooth_morozov_iter(spectral_data, noise,
#     #                                   lmbd_min=0.005, lmbd_step=0.01,
#     #                                   lmbd_max=1.e2, rel_lmbd=5e-4,
#     #                                   res_steps=2, d=3)
#     out = whittaker_smooth_mscr(spectral_data, noise,
#                                 lmbd_init=1.e10, d=2, tau=5., min_k=2,
#                                 q=0.2)
#     smooth_data = out[0]

#     # plt.subplot(211)
#     # plt.plot(model[:, y, x], linewidth=4, alpha=0.5, color='gray')
#     # plt.plot(spectral_data)
#     # plt.plot(smooth_data)
#     # plt.subplot(212)
#     # plt.semilogy(out[1])
#     # plt.draw()
#     # input("{0}, {1}?".format(y, x))
#     # plt.clf()
#     # continue


#     # smooth_data, opt_out = \
#     #     whittaker_smooth_gcv(spectral_data, 100., d=3)
#     # print("Smoothing scales {}".format(10**opt_out['x']))
#     # print("Smoothing scales {}".format(opt_out))

#     # lambdas[y, x] = opt_out
#     # continue

#     # smooth_data = spectral_data

#     # # Get regularized derivatives
#     # u1, u2, u3, u4 = reg_derivs(vels, smooth_data,
#     #                             alpha=a1,
#     #                             # mode='python')
#     #                             mode='python_adapt',
#     #                             max_iter=100,
#     #                             # K=2 * (noise / np.sqrt(cumul_smooth)) / np.max(smooth_data))
#     #                             K=2 * noise / np.max(smooth_data))

#     # u1 = np.gradient(model[:, y, x])
#     u1 = np.gradient(smooth_data)
#     u2 = np.gradient(u1)
#     u3 = np.gradient(u2)
#     u4 = np.gradient(u3)

#     test = triple_point_find(vels, spectral_data, smooth_data,
#                              u1, u2, u3, u4,
#                              noise,
#                              verbose=True, thresh=5.,
#                              size_scale=3.,
#                              deconv_filt=False,
#                              fix_indept_peak_width=False)

#     print(test)

#     # plt.subplot(231)
#     plt.subplot(311)
#     plt.plot(vels, spectral_data)
#     plt.plot(vels, smooth_data, '-', linewidth=4, alpha=1.,
#              color='r')
#     plt.plot(vels, smooth_data, ':', linewidth=2, alpha=0.7,
#              color='k')
#     plt.plot(vels, model[:, y, x],
#              color='gray', alpha=0.5, linewidth=3)
#     plt.plot(vels, model1[:, y, x], '--',
#              color='gray', alpha=0.5, linewidth=3)
#     plt.plot(vels, model2[:, y, x], '--',
#              color='gray', alpha=0.5, linewidth=3)
#     plt.plot(vels, model3[:, y, x], '--',
#              color='gray', alpha=0.5, linewidth=3)

#     # for trip in test[1]['trip_pts']:
#     #         plt.axvline(trip[0], color='b', linestyle='--')
#     #         plt.axvline(trip[2], color='b', linestyle=':')
#     #         plt.axvline(trip[1], color='r', linestyle='-')

#     # for i in range(len(test[0]['amp'])):
#     #     plt.plot(vels, gaussian(vels, test[0]['amp'][i],
#     #                             test[0]['mean'][i],
#     #                             test[0]['sigma'][i]))

#     # plt.subplot(232)
#     # # plt.plot(vels, u1)
#     # # for trip in test[1]['trip_pts']:
#     # #         plt.axvline(trip[0], color='b', linestyle='--')
#     # #         plt.axvline(trip[2], color='b', linestyle=':')
#     # #         plt.axvline(trip[1], color='r', linestyle='-')

#     # # plt.subplot(233)
#     plt.subplot(312)
#     plt.plot(vels, u2)
#     plt.plot(vels, np.gradient(np.gradient(model[:, y, x])),
#              color='gray', alpha=0.5, linewidth=3)
#     for trip in test[1]['trip_pts']:
#             plt.axvline(trip[0], color='b', linestyle='--')
#             plt.axvline(trip[2], color='b', linestyle=':')
#             plt.axvline(trip[1], color='r', linestyle='-')

#     # plt.subplot(234)
#     # plt.plot(vels, u3)
#     # for trip in test[1]['trip_pts']:
#     #         plt.axvline(trip[0], color='b', linestyle='--')
#     #         plt.axvline(trip[2], color='b', linestyle=':')
#     #         plt.axvline(trip[1], color='r', linestyle='-')

#     # plt.subplot(235)
#     # plt.plot(vels, u4)
#     # for trip in test[1]['trip_pts']:
#     #         plt.axvline(trip[0], color='b', linestyle='--')
#     #         plt.axvline(trip[2], color='b', linestyle=':')
#     #         plt.axvline(trip[1], color='r', linestyle='-')


#     # plt.subplot(313)
#     # plt.semilogy(out[1])

#     plt.draw()

#     print("Actual parameters:")
#     print(params1[y, x])
#     print(params2[y, x])
#     print(params3[y, x])

#     input("{0} {1}".format(y, x))
#     pl.clf()

# print(argh)

# Add noise to the initial parameters for the fit
noisy_params = params.copy()
noisy_params[..., 0] += np.random.normal(0, amp / 10., noisy_params[..., 0].shape)
noisy_params[..., 0][noisy_params[..., 0] < 0] = \
    np.abs(np.random.normal(0, amp / 10., np.sum(noisy_params[..., 0] < 0)))
noisy_params[..., 1] += np.random.normal(0, std / 2., noisy_params[..., 1].shape)
noisy_params[..., 2] += np.random.normal(0, std / 8., noisy_params[..., 2].shape)
# Just set negative widths to the actual value
noisy_params[..., 2][noisy_params[..., 2] < 0] = \
    params[..., 2][noisy_params[..., 2] < 0]

noisy_params[..., 3] += np.random.normal(0, amp2 / 10., noisy_params[..., 3].shape)
noisy_params[..., 3][noisy_params[..., 3] < 0] = \
    np.abs(np.random.normal(0, amp2 / 10., np.sum(noisy_params[..., 3] < 0)))

noisy_params[..., 4] += np.random.normal(0, std2 / 2., noisy_params[..., 4].shape)
noisy_params[..., 5] += np.random.normal(0, std2 / 8.,
                                         noisy_params[..., 5].shape)
# Just set negative widths to the actual value
noisy_params[..., 5][noisy_params[..., 5] < 0] = \
    params[..., 5][noisy_params[..., 5] < 0]

# noisy_params[..., 6] += np.random.normal(0, amp3 / 10., noisy_params[..., 6].shape)
# noisy_params[..., 6][noisy_params[..., 6] < 0] = \
#     np.abs(np.random.normal(0, amp3 / 10., np.sum(noisy_params[..., 6] < 0)))

# noisy_params[..., 7] += np.random.normal(0, std3 / 2., noisy_params[..., 7].shape)
# noisy_params[..., 8] += np.random.normal(0, std3 / 8.,
#                                          noisy_params[..., 8].shape)
# # Just set negative widths to the actual value
# noisy_params[..., 8][noisy_params[..., 8] < 0] = \
#     params[..., 8][noisy_params[..., 8] < 0]

# Try using observational estimate to see how close we need initial guesses
# tpeak = noisy_model.max(0)
# cent = noisy_model.argmax(0)
# lwidth = np.ones_like(tpeak) * 5.

# noisy_params = np.zeros_like(params)

# for i in range(ncomps):
#     noisy_params[..., 3 * i] = ((tpeak / float(ncomps)) + np.random.normal(0., noise, tpeak.shape))
#     noisy_params[..., 3 * i + 1] = (cent + np.random.normal(0., 1., cent.shape))
#     noisy_params[..., 3 * i + 2] = (lwidth + np.random.normal(0., 1., lwidth.shape))


cube_spat_size = model[0].shape


# Make a velocity cube to avoid tiling this axis in each fcn call
spat_shape = (npix, npix)
vels_cube = np.tile(vels, spat_shape + (1,)).swapaxes(0, 2).astype(np.float64)

# For the convolution approach, make the 2nd deriv of a gaussian kernel
# from astropy.convolution.kernels import Gaussian2DKernel

# kern = Gaussian2DKernel(neigb_scale)
# grad1 = np.gradient(kern.array)
# grad_mag = np.sqrt(grad1[0]**2 + grad1[1]**2)
# grad2 = np.gradient(grad_mag)
# grad_diff_mag = np.sqrt(grad2[0]**2 + grad2[1]**2)

param_weights = [1., 1., 10.]


def loglike(pars):
    pars = pars.reshape(cube_spat_size + (3 * ncomps,))

    # return spatial_reg_cube_model_conv_cyth(vels_cube, noisy_model, model_errs,
    #                                         # pars, kern.array, gaussian,
    #                                         pars, grad_diff_mag, gaussian,
    #                                         vel_surf=params[..., 1],
    #                                         param_weights=[1., 10., 1.],)

    return spatial_reg_cube_model_cyth(vels_cube, noisy_model, model_errs,
                                       pars, neigb_scale, gaussian,
                                       param_weights=param_weights,)
                                       # vel_surf=params[..., 1],)

    # return spatial_reg_cube_model(vels_cube, noisy_model, model_errs,
    #                               pars, neigb_scale,
    #                               use_reg=True,
    #                               # vel_surf=params[..., 1],
    #                               param_weights=[0.5, 1., 10.],)
    #                               # loss_func=np.arctan)
    #                               # vel_surf=params[..., 2])

    # return spatial_reg_cube_model_onoff(vels, noisy_model, model_errs,
    #                                     pars, neigb_scale,
    #                                     use_reg=True,
    #                                     vel_surf=params[..., 1],
    #                                     param_weights=[1., 10., 10.],
    #                                     # loss_func=np.arctan,
    #                                     null_model_thresh=noise)


def jacobian(pars):
    pars = pars.reshape(cube_spat_size + (3 * ncomps,))

    # return gaussian_jacobian(vels_cube, noisy_model, pars) + \
    return gaussian_jacobian(vels_cube, noisy_model, pars, model_errs) + \
        spatial_reg_jacobian(pars, neigb_scale,
                             param_weights=param_weights,)

t2 = time.time()

# The non-reg whole cube fit can be done per-spectrum (probably faster)
yy, xx = np.indices(model.shape[1:])

fit_params_noreg = np.zeros_like(noisy_params)

bounds_noreg = tuple([(0., None), (vels.min(), vels.max()),
                      (1., None)] * ncomps)

for ii, (y, x) in enumerate(zip(yy.ravel(), xx.ravel())):

    def loglike_perspec(pars):
        for j in range(pars.size // 3):
            if j == 0:
                mod = gaussian(vels, pars[3 * j],
                               pars[3 * j + 1], pars[3 * j + 2])
            else:
                mod += gaussian(vels, pars[3 * j],
                                pars[3 * j + 1], pars[3 * j + 2])

        return np.sum((noisy_model[:, y, x] - mod)**2)

    output_noreg = minimize(loglike_perspec, noisy_params[y, x, :],
                            bounds=bounds_noreg)

    fit_params_noreg[y, x, :] = output_noreg['x']

t3 = time.time()

t0 = time.time()
model_bounds = generate_bounds(noisy_params, vels.min(), vels.max(), sigma_down=1.)

from scipy.optimize import BFGS, Bounds

# output = minimize(loglike, fit_params_noreg.ravel(),
# output = minimize(loglike, noisy_params.ravel(),
#                   method='trust-constr',
#                   jac=jacobian,
#                   hess=BFGS(exception_strategy='skip_update'),
#                   bounds=Bounds(model_bounds[0], model_bounds[1]),
#                   # tol=1e-4 / noise**2,
#                   tol=1e-4,
#                   options=dict(verbose=3))
#                                 # , xtol=1.e0, maxiter=20,))
#                                # factorization_method='NormalEquation'))

output = minimize(loglike, noisy_params.ravel(),
                  method='L-BFGS-B',
                  jac=jacobian,
                  # hess=BFGS(exception_strategy='skip_update'),
                  bounds=Bounds(model_bounds[0], model_bounds[1]),
                  # tol=1e0,
                  options=dict(disp=3))#, maxiter=1000, maxfev=50000,
                               # ftol=1e-4, gtol=1e-6))

# minimizer_kwargs = dict(method="L-BFGS-B", bounds=Bounds(model_bounds[0], model_bounds[1]))
# minimizer_kwargs = dict(method="trust-constr",
#                         bounds=Bounds(model_bounds[0], model_bounds[1]),
#                         jac=jacobian,
#                         hess=BFGS(exception_strategy='skip_update'))

# # output = basinhopping(loglike, noisy_params.ravel(),
# output = basinhopping(loglike, params.ravel(),
#                       minimizer_kwargs=minimizer_kwargs,
#                       disp=True, niter_success=None,
#                       niter=1000)

                  # options=dict(disp=3, maxiter=50, maxfev=30000,
                  #              ftol=1e-4, gtol=1e-6))

# output = least_squares(loglike, noisy_params.ravel(), method='trf',
#                        tr_solver='lsmr', verbose=2,  # ftol=1e-2,
#                        bounds=model_bounds,
#                        max_nfev=150)
fit_params = output['x'].reshape((npix, npix, 3 * ncomps))
t1 = time.time()

print("Reg fit time: {}".format(t1 - t0))
print("No Reg fit time: {}".format(t3 - t2))

# Parameter uncertainties
# uncerts = np.zeros(len(output.x))
# ftol = 2.220446049250313e-09
# hess_inv = output.hess_inv.todense()
# for i in range(len(output.x)):
#     uncerts[i] = np.sqrt(max(1, abs(output.fun)) * ftol * hess_inv[i, i])
#     print('{0:12.4e} ± {1:.1e}'.format(output.x[i], uncerts[i]))

lstyles = ['--', '-.', ':']

pl.figure(1).clf()
fig, axes = pl.subplots(npix, npix, sharex=True, sharey=True, num=1)

for ii, ((yy, xx), ax) in enumerate(zip(np.ndindex((npix, npix)),
                                        axes.ravel())):
    ax.plot(model[:, yy, xx], 'k--', alpha=0.6, zorder=-10, linewidth=5,
            drawstyle='steps-mid')
    ax.plot(model1[:, yy, xx], 'k' + lstyles[0], alpha=0.5, zorder=-10,
            linewidth=2,
            drawstyle='steps-mid')
    ax.plot(model2[:, yy, xx], 'k' + lstyles[1], alpha=0.5, zorder=-10,
            linewidth=2,
            drawstyle='steps-mid')
    ax.plot(model3[:, yy, xx], 'k' + lstyles[2], alpha=0.5, zorder=-10,
            linewidth=2,
            drawstyle='steps-mid')
    ax.plot(noisy_model[:, yy, xx], 'k-', zorder=-5, linewidth=2,
            drawstyle='steps-mid')
    # ax.plot(gaussian(vels, fit_params_noreg[yy, xx, 0],
    #                  fit_params_noreg[yy, xx, 1],
    #                  fit_params_noreg[yy, xx, 2]),
    #         'b--', zorder=0, linewidth=1,
    #         drawstyle='steps-mid')

    total = np.zeros_like(model[:, yy, xx])
    total_noreg = np.zeros_like(model[:, yy, xx])
    for i in range(ncomps):

        mod = gaussian(vels, fit_params[yy, xx, 3 * i],
                       # * (fit_params[yy, xx, 3 * i] > noise),
                       fit_params[yy, xx, 3 * i + 1],
                       fit_params[yy, xx, 3 * i + 2])

        mod_noreg = gaussian(vels, fit_params_noreg[yy, xx, 3 * i],
                             fit_params_noreg[yy, xx, 3 * i + 1],
                             fit_params_noreg[yy, xx, 3 * i + 2])

        total += mod
        total_noreg += mod_noreg

        ax.plot(mod,
                'b' + lstyles[i], zorder=0, linewidth=1,
                drawstyle='steps-mid')
        # ax.plot(mod_noreg,
        #         'r' + lstyles[i], zorder=0, linewidth=1,
        #         drawstyle='steps-mid')

    ax.plot(total,
            'b-', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    # ax.plot(total_noreg,
    #         'r-', zorder=0, linewidth=1,
    #         drawstyle='steps-mid')


pl.tight_layout()
pl.subplots_adjust(hspace=0, wspace=0)

for i in range(ncomps):

    pl.figure()

    pl.subplot(321)
    pl.imshow(params[..., 3*i] - fit_params[..., 3*i])
    pl.title("Regularized")
    pl.colorbar()
    pl.subplot(323)
    pl.imshow(params[..., 3*i + 1] - fit_params[..., 3*i + 1])
    pl.colorbar()
    pl.subplot(325)
    pl.imshow(params[..., 3*i + 2] - fit_params[..., 3*i + 2])
    pl.colorbar()
    pl.subplot(322)
    pl.imshow(params[..., 3*i] - fit_params_noreg[..., 3*i])
    pl.title("Indep't")
    pl.colorbar()
    pl.subplot(324)
    pl.imshow(params[..., 3*i + 1] - fit_params_noreg[..., 3*i + 1])
    pl.colorbar()
    pl.subplot(326)
    pl.imshow(params[..., 3*i + 2] - fit_params_noreg[..., 3*i + 2])
    pl.colorbar()

    pl.figure()

    pl.subplot(331)
    pl.imshow(params[..., 3*i], origin='lower',
              vmin=params[..., 3*i].min(),
              vmax=params[..., 3*i].max())
    pl.title("Actual")
    pl.colorbar()
    pl.subplot(334)
    pl.imshow(params[..., 3*i + 1], origin='lower',
              vmin=params[..., 3*i + 1].min(),
              vmax=params[..., 3*i + 1].max())
    pl.colorbar()
    pl.subplot(337)
    pl.imshow(params[..., 3*i + 2], origin='lower',)
              # vmin=params[..., 3*i + 2].min(),
              # vmax=params[..., 3*i + 2].max())
    pl.colorbar()


    pl.subplot(332)
    pl.imshow(noisy_params[..., 3*i], origin='lower',
              vmin=params[..., 3*i].min(),
              vmax=params[..., 3*i].max())
    pl.title("Initial")
    pl.colorbar()
    pl.subplot(335)
    pl.imshow(noisy_params[..., 3*i + 1], origin='lower',
              vmin=params[..., 3*i + 1].min(),
              vmax=params[..., 3*i + 1].max())
    pl.colorbar()
    pl.subplot(338)
    pl.imshow(noisy_params[..., 3*i + 2], origin='lower')
              # vmin=params[..., 3*i + 2].min(),
              # vmax=params[..., 3*i + 2].max())
    pl.colorbar()

    pl.subplot(333)
    pl.imshow(fit_params[..., 3*i], origin='lower',
              vmin=params[..., 3*i].min(),
              vmax=params[..., 3*i].max())
    pl.title("Regularized")
    pl.colorbar()
    pl.subplot(336)
    pl.imshow(fit_params[..., 3*i + 1], origin='lower',
              vmin=params[..., 3*i + 1].min(),
              vmax=params[..., 3*i + 1].max())
    pl.colorbar()
    pl.subplot(339)
    pl.imshow(fit_params[..., 3*i + 2], origin='lower',)
              # vmin=params[..., 3*i + 2].min(),
              # vmax=params[..., 3*i + 2].max())

    pl.colorbar()
