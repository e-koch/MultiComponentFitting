
cimport cython
import numpy as np
cimport numpy as np
from scipy.signal import convolve2d

from libc.math cimport sqrt

cdef double pi = np.pi
cdef double SQRT_2PI = sqrt(2 * pi)


def model_residual(np.ndarray[np.float64_t, ndim=3] vels,
                   np.ndarray[np.float64_t, ndim=3] cube,
                   np.ndarray[np.float64_t, ndim=3] params,
                   model):

    cdef np.ndarray[np.float64_t, ndim=3] model_eval
    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3

    for j in range(ncomps):

        if j == 0:
            model_eval = model(vels, params[..., 3 * j],
                               params[..., 3 * j + 1],
                               params[..., 3 * j + 2],)
        else:
            model_eval += model(vels, params[..., 3 * j],
                                params[..., 3 * j + 1],
                                params[..., 3 * j + 2],)

    cdef np.ndarray[np.float64_t, ndim=3] residuals = model_eval - cube

    # When a std err map is given:
    # cdef np.ndarray[np.float64_t, ndim=3] residuals = (model_eval - cube) / std_err**2

    return residuals


def gaussian_cy(xax, amp, cen, wid):
    return np.exp(-(xax - cen)**2 / (2 * wid**2)) * amp


def gaussian_part_derivs(np.ndarray[np.float64_t, ndim=3] vels,
                         np.ndarray[np.float64_t, ndim=3] params,):

    # Make the output cube. Npars by len(vels)

    cdef int j, y, x

    cdef int vshape = vels.shape[0]
    cdef int yshape = vels.shape[1]
    cdef int xshape = vels.shape[2]

    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3
    cdef int total_par = yshape * xshape * zpar

    cdef np.ndarray[np.float64_t, ndim=2] grad_arr = np.zeros((vshape, total_par), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] gauss_part
    cdef np.ndarray[np.float64_t, ndim=1] pars

    yy, xx = np.indices((yshape, xshape))

    cdef int i = 0

    for y, x in zip(yy.ravel(), xx.ravel()):

        for j in range(ncomps):
            pars = params[y, x, 3 * j: 3 * j + 3]

            gauss_part = np.exp(- (vels[:, 0, 0] - pars[1])**2 / (2 * pars[2]**2))

            # Deriv of amp
            grad_arr[:, i] = gauss_part
            i += 1

            # Deriv of centre
            grad_arr[:, i] = gauss_part * pars[0] * (vels[:, 0, 0] - pars[1]) / (pars[2]**2)
            i += 1

            # Deriv of sigma
            grad_arr[:, i] = gauss_part * pars[0] * (vels[:, 0, 0] - pars[1])**2 / (pars[2]**3)
            i += 1

    return grad_arr


def gaussian_jacobian(np.ndarray[np.float64_t, ndim=3] vels,
                      np.ndarray[np.float64_t, ndim=3] cube,
                      np.ndarray[np.float64_t, ndim=3] params,
                      cube_err=None):

    # Eventually add a stderr map as optional here

    cdef np.ndarray[np.float64_t, ndim=2] gauss_grads = gaussian_part_derivs(vels, params)

    cdef np.ndarray[np.float64_t, ndim=3] residuals = model_residual(vels, cube, params, gaussian_cy)

    cdef int j, k, y, x

    cdef int vshape = vels.shape[0]
    cdef int yshape = vels.shape[1]
    cdef int xshape = vels.shape[2]

    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3

    cdef int total_par = yshape * xshape * zpar

    cdef np.ndarray[np.float64_t, ndim=1] obj_grads = np.zeros((total_par,))

    cdef int i = 0

    yy, xx = np.indices((yshape, xshape))

    for y, x in zip(yy.ravel(), xx.ravel()):

        for j in range(ncomps):

            for k in range(3):
                if cube_err is not None:
                    obj_grads[i] = np.sum(gauss_grads[:, i] * (residuals[:, y, x] / cube_err[y, x]**2.))
                else:
                    obj_grads[i] = np.sum(gauss_grads[:, i] * residuals[:, y, x])
                i += 1

    return obj_grads


def spatial_reg_jacobian(np.ndarray[np.float64_t, ndim=3] params,
                         double spatial_scale,
                         param_weights=[0.2, 4., 1.],
                         vel_surf=None):

    cdef int j, k, y, x

    cdef int yshape = params.shape[0]
    cdef int xshape = params.shape[1]

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3
    cdef int total_par = yshape * xshape * zpar

    cdef np.ndarray[np.float64_t, ndim=1] obj_grads = np.zeros((total_par,))

    # Properties of the spatial kernel
    cdef double spat_sq = spatial_scale**2
    cdef double norm = 1 / (2 * np.pi * spat_sq)

    yy, xx = np.indices((yshape, xshape))

    cdef np.ndarray[np.float64_t, ndim=1] yy_f = yy.ravel().astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] xx_f = xx.ravel().astype(np.float64)

    cdef double vel_surf_diff, dist_weight

    cdef int i = 0

    for y, x in zip(yy.ravel(), xx.ravel()):

        all_dists = (yy_f - y)**2 + (xx_f - x)**2

        neighbs = np.where(all_dists < spat_sq)

        for j in range(ncomps):
            pars = params[y, x, 3 * j: 3 * j + 3]

            for y2, x2, dist in zip(yy.ravel()[neighbs],
                                    xx.ravel()[neighbs],
                                    all_dists[neighbs]):

                par_diff = pars - params[y2, x2, 3 * j: 3 * j + 3]

                if vel_surf is not None:
                    vel_surf_diff = \
                        np.abs(vel_surf[y, x] - vel_surf[y2, x2])
                    if np.abs(par_diff[1]) < vel_surf_diff:
                        par_diff[1] = 0.

                dist_weight = norm * np.exp(- 0.5 * dist / spat_sq)

                # err_weight = 1.  # cube_err[y, x] * cube_err[y2, x2]

                # Since W is diagonal, could just multiply arrays with
                # diagonal. But this is the general form for now
                # lnlike += err_weight * dist_weight * (par_diff[0] * param_weights[0])**2
                # lnlike += err_weight * dist_weight * (par_diff[1] * param_weights[1])**2
                # lnlike += err_weight * dist_weight * (par_diff[2] * param_weights[2])**2
                obj_grads[i] += dist_weight * (par_diff[0] * param_weights[0]**2)
                obj_grads[i + 1] += dist_weight * (par_diff[1] * param_weights[1]**2)
                obj_grads[i + 2] += dist_weight * (par_diff[2] * param_weights[2]**2)

            i += 3

    return obj_grads


def spatial_reg_cube_model_cyth(np.ndarray[np.float64_t, ndim=3] vels,
                                np.ndarray[np.float64_t, ndim=3] cube,
                                np.ndarray[np.float64_t, ndim=2] cube_err,
                                np.ndarray[np.float64_t, ndim=3] params,
                                double spatial_scale,
                                model,
                                vel_surf=None,
                                param_weights=[0.2, 4., 1.]):

    cdef int j, k, y, x

    cdef np.ndarray[np.float64_t, ndim=1] all_dists, par_diff
    cdef np.ndarray[np.int_t, ndim=2, cast=True] yy, xx

    cdef int yshape = cube.shape[1]
    cdef int xshape = cube.shape[2]

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3

    # Vels should be a cube with matching dimensions to the cube
    # This way we can perform cube operations to save time.

    cdef double lnlike = 0.

    # err_sq = cube_err**2

    # Properties of the spatial kernel
    cdef double spat_sq = spatial_scale**2
    cdef double norm = 1 / (2 * np.pi * spat_sq)

    residuals = model_residual(vels, cube, params, model)

    lnlike = np.sum((residuals / cube_err)**2)

    # Loop through spatial positions
    yy, xx = np.indices((yshape, xshape))

    cdef int i = 0

    cdef np.ndarray[np.float64_t, ndim=1] yy_f = yy.ravel().astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] xx_f = xx.ravel().astype(np.float64)

    for y, x in zip(yy.ravel(), xx.ravel()):

        # Only include regions whose distance is within the spatial scale
        # Exclude double counting neighbours by excluding pixels already
        # before y, x in yy and xx.

        all_dists = (yy_f - y)**2 + (xx_f - x)**2

        neighbs = np.where(all_dists[i + 1:] < spat_sq)

        for j in range(ncomps):
            pars = params[y, x, 3 * j: 3 * j + 3]

            # Add a penalization on the amplitude to
            # force small amplitudes to 0.
            # lambdaa = 1.

            # Impose a logistic function, normalized by some limit
            # z = pars[0]  # / cube_err[y, x]
            # z0 = 3 * cube_err[y, x]
            # k = 1.
            # amp_penalty = 1 - 1. / (1 + np.exp(-k * (z - z0)**2))

            # lnlike += amp_penalty * lambdaa  # * cube_err[y, x]**2

            for y2, x2, dist in zip(yy.ravel()[i + 1:][neighbs],
                                    xx.ravel()[i + 1:][neighbs],
                                    all_dists[i + 1:][neighbs]):

                par_diff = pars - params[y2, x2, 3 * j: 3 * j + 3]

                if vel_surf is not None:
                    vel_surf_diff = \
                        np.abs(vel_surf[y, x] - vel_surf[y2, x2])
                    if np.abs(par_diff[1]) < vel_surf_diff:
                        par_diff[1] = 0.

                dist_weight = norm * np.exp(- 0.5 * dist / spat_sq)

                # err_weight = 1.  # cube_err[y, x] * cube_err[y2, x2]

                # Since W is diagonal, could just multiply arrays with
                # diagonal. But this is the general form for now
                # lnlike += err_weight * dist_weight * (par_diff[0] * param_weights[0])**2
                # lnlike += err_weight * dist_weight * (par_diff[1] * param_weights[1])**2
                # lnlike += err_weight * dist_weight * (par_diff[2] * param_weights[2])**2
                lnlike += dist_weight * (par_diff[0] * param_weights[0])**2
                lnlike += dist_weight * (par_diff[1] * param_weights[1])**2
                lnlike += dist_weight * (par_diff[2] * param_weights[2])**2

        i += 1

    return lnlike


def spatial_reg_cube_jac_cyth(np.ndarray[np.float64_t, ndim=3] vels,
                              np.ndarray[np.float64_t, ndim=3] cube,
                              np.ndarray[np.float64_t, ndim=2] cube_err,
                              np.ndarray[np.float64_t, ndim=3] params,
                              double spatial_scale,
                              model,
                              vel_surf=None,
                              param_weights=[0.2, 4., 1.]):

    cdef int j, k, y, x

    cdef np.ndarray[np.float64_t, ndim=3] model_eval
    cdef np.ndarray[np.float64_t, ndim=1] all_dists, par_diff
    cdef np.ndarray[np.int_t, ndim=2, cast=True] yy, xx

    cdef int yshape = cube.shape[1]
    cdef int xshape = cube.shape[2]

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    cdef int zpar = params.shape[2]
    cdef int ncomps = zpar / 3

    # Vels should be a cube with matching dimensions to the cube
    # This way we can perform cube operations to save time.

    cdef double lnlike = 0.

    # err_sq = cube_err**2

    # Properties of the spatial kernel
    cdef double spat_sq = spatial_scale**2
    cdef double norm = 1 / (2 * np.pi * spat_sq)

    residuals = model_residual(vels, cube, params, model)

    lnlike = np.sum((residuals / cube_err)**2)
    # lnlike = np.sum((model_eval - cube)**2)

    # Loop through spatial positions
    yy, xx = np.indices((yshape, xshape))

    cdef int i = 0

    cdef np.ndarray[np.float64_t, ndim=1] yy_f = yy.ravel().astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] xx_f = xx.ravel().astype(np.float64)

    for y, x in zip(yy.ravel(), xx.ravel()):

        # Only include regions whose distance is within the spatial scale
        # Exclude double counting neighbours by excluding pixels already
        # before y, x in yy and xx.

        all_dists = (yy_f - y)**2 + (xx_f - x)**2

        neighbs = np.where(all_dists[i + 1:] < spat_sq)

        for j in range(ncomps):
            pars = params[y, x, 3 * j: 3 * j + 3]

            # Add a penalization on the amplitude to
            # force small amplitudes to 0.
            # lambdaa = 1.

            # Impose a logistic function, normalized by some limit
            # z = pars[0]  # / cube_err[y, x]
            # z0 = 3 * cube_err[y, x]
            # k = 1.
            # amp_penalty = 1 - 1. / (1 + np.exp(-k * (z - z0)**2))

            # lnlike += amp_penalty * lambdaa  # * cube_err[y, x]**2

            for y2, x2, dist in zip(yy.ravel()[i + 1:][neighbs],
                                    xx.ravel()[i + 1:][neighbs],
                                    all_dists[i + 1:][neighbs]):

                par_diff = pars - params[y2, x2, 3 * j: 3 * j + 3]

                if vel_surf is not None:
                    vel_surf_diff = \
                        np.abs(vel_surf[y, x] - vel_surf[y2, x2])
                    if np.abs(par_diff[1]) < vel_surf_diff:
                        par_diff[1] = 0.

                dist_weight = norm * np.exp(- 0.5 * dist / spat_sq)

                # err_weight = 1.  # cube_err[y, x] * cube_err[y2, x2]

                # Since W is diagonal, could just multiply arrays with
                # diagonal. But this is the general form for now
                # lnlike += err_weight * dist_weight * (par_diff[0] * param_weights[0])**2
                # lnlike += err_weight * dist_weight * (par_diff[1] * param_weights[1])**2
                # lnlike += err_weight * dist_weight * (par_diff[2] * param_weights[2])**2
                lnlike += dist_weight * (par_diff[0] * param_weights[0])**2
                lnlike += dist_weight * (par_diff[1] * param_weights[1])**2
                lnlike += dist_weight * (par_diff[2] * param_weights[2])**2

        i += 1

    return lnlike


def spatial_reg_cube_model_conv_cyth(np.ndarray[np.float64_t, ndim=3] vels,
                                     np.ndarray[np.float64_t, ndim=3] cube,
                                     np.ndarray[np.float64_t, ndim=2] cube_err,
                                     np.ndarray[np.float64_t, ndim=3] params,
                                     np.ndarray[np.float64_t, ndim=2] spat_diff_kern,
                                     model,
                                     vel_surf=None,
                                     param_weights=[0.2, 4., 1.]):

    cdef int j, k, ncomps, y, x

    cdef np.ndarray[np.float64_t, ndim=3] model_eval
    cdef np.ndarray[np.float64_t, ndim=1] all_dists, par_diff
    cdef np.ndarray[np.int_t, ndim=2, cast=True] yy, xx

    cdef int yshape = cube.shape[1]
    cdef int xshape = cube.shape[2]

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    cdef int zpar = params.shape[2]
    ncomps = zpar / 3

    # Vels should be a cube with matching dimensions to the cube
    # This way we can perform cube operations to save time.

    cdef double lnlike = 0.

    # err_sq = cube_err**2

    # Tile the velocities to evaluate the whole model
    residuals = model_residual(vels, cube, params, model)


    lnlike = np.sum((residuals / cube_err)**2)

    # Convolve the maps by the spatial kernel for use in
    # regularization terms
    for j in range(ncomps):
        pars = params[..., 3 * j: 3 * j + 3]

        for k in range(3):

            if vel_surf is not None and k == 1:
                # Allow subtracting a large-scale velocity field
                # conv_diff = convolve2d(pars[..., k] - vel_surf, spat_diff_kern, mode='same')
                conv_diff = convolve2d(pars[..., k] - vel_surf, spat_diff_kern, mode='same') - pars[..., k]

            else:
                # conv_diff = convolve2d(pars[..., k], spat_diff_kern, mode='same')
                conv_diff = convolve2d(pars[..., k], spat_diff_kern, mode='same') - pars[..., k]

            lnlike += np.sum((conv_diff * param_weights[k])**2)

    return lnlike
