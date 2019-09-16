
'''
Define a spatially-regularized model.
'''

import numpy as np


def gaussian(xax, amp, cen, wid):
    return np.exp(-(xax - cen)**2 / (2 * wid**2)) * amp


def make_neighborhood(xx, yy, scale_size):
    W = np.zeros_like((xx.size, xx.size))

    norm = 1 / (2 * np.pi * scale_size)

    scale_sq = scale_size**2

    for i in range(xx.size):
        for j in range(i):
            dist_diff_sq = (xx.ravel()[i] - xx.ravel()[j])**2 + \
                (yy.ravel()[i] - yy.ravel()[j])**2
            W[i, j] = norm * np.exp(-dist_diff_sq / (2 * scale_sq))

    W = W + W.T + np.eye(xx.size)

    return W


def spatial_reg_cube_model(vels, cube, cube_err, params, spatial_scale,
                           model=gaussian, use_reg=True, vel_surf=None,
                           param_weights=[0.2, 4., 1.],
                           loss_func=lambda x: x):

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    ncomps = params.shape[-1] // 3
    spat_shape = cube.shape[1:]

    # Vels should be a cube with matching dimensions to the cube
    # This way we can perform cube operations to save time.

    lnlike = 0.

    # err_sq = cube_err**2

    # Properties of the spatial kernel
    spat_sq = spatial_scale**2
    norm = 1 / (2 * np.pi * spat_sq)

    # Tile the velocities to evaluate the whole model
    for j in range(ncomps):

        if j == 0:
            model_eval = model(vels, params[..., 3 * j],
                               params[..., 3 * j + 1],
                               params[..., 3 * j + 2],)
        else:
            model_eval += model(vels, params[..., 3 * j],
                                params[..., 3 * j + 1],
                                params[..., 3 * j + 2],)


    lnlike = np.sum(loss_func(((model_eval - cube) / cube_err)**2))

    # Loop through spatial positions
    yy, xx = np.indices(cube.shape[1:])

    if use_reg:
        for i, (y, x) in enumerate(zip(yy.ravel(), xx.ravel())):

            # mod_resids = np.sqrt(np.sum(loss_func((model_yx - cube[:, y, x])**2))) / \
                # cube_err[y, x]
            # mod_resids = np.sum(loss_func((model_yx - cube[:, y, x])**2)) / \
            #     cube_err[y, x]**2

            # lnlike += resids

            # Regularize vs. params

            # Only include regions whose distance is within the spatial scale
            # Exclude double counting neighbours by excluding pixels already
            # before y, x in yy and xx.
            all_dists = np.sqrt((yy.ravel() - y)**2 + (xx.ravel() - x)**2)
            neighbs = np.where(all_dists[i + 1:] < spatial_scale)

            for j in range(ncomps):
                pars = params[y, x, 3 * j: 3 * j + 3]

                # Add a penalization on the amplitude to
                # force small amplitudes to 0.
                lambdaa = 1.

                # Impose a logistic function, normalized by some limit
                # z = pars[0]  # / cube_err[y, x]
                # z0 = 3 * cube_err[y, x]
                # k = 1.
                # amp_penalty = 1 - 1. / (1 + np.exp(-k * (z - z0)**2))

                # lnlike += amp_penalty * lambdaa  # * cube_err[y, x]**2

                for y2, x2 in zip(yy.ravel()[i + 1:][neighbs],
                                  xx.ravel()[i + 1:][neighbs]):

                    par_diff = pars - params[y2, x2, 3 * j: 3 * j + 3]

                    dist = (x2 - x)**2 + (y2 - y)**2

                    if vel_surf is not None:
                        vel_surf_diff = \
                            np.abs(vel_surf[y, x] - vel_surf[y2, x2])
                        if np.abs(par_diff[1]) < vel_surf_diff:
                            par_diff[1] = 0.

                    dist_weight = norm * np.exp(- 0.5 * dist / spat_sq)

                    err_weight = 1.  # cube_err[y, x] * cube_err[y2, x2]

                    # Since W is diagonal, could just multiply arrays with
                    # diagonal. But this is the general form for now
                    # lnlike += err_weight * dist_weight * \
                    #     np.sqrt(sum([par_diff[k]**2 * param_weights[k] for k in
                    #                  range(3)]))
                    lnlike += err_weight * dist_weight * \
                        sum([par_diff[k]**2 * param_weights[k] for k in
                             range(3)])

    return lnlike


def spatial_reg_cube_model_onoff(vels, cube, cube_err, params, spatial_scale,
                                 model=gaussian, use_reg=True, vel_surf=None,
                                 param_weights=[0.2, 4., 1.],
                                 null_model_thresh=0.,
                                 loss_func=lambda x: x):

    # params should have the same spatial shape with 3 in the next dimension
    # for each Gaussian parameter

    ncomps = params.shape[-1] // 3

    # spec_axis = cube.spectral_axis.value
    spec_axis = vels

    lnlike = 0.

    # err_sq = cube_err**2

    # Properties of the spatial kernel
    spat_sq = spatial_scale**2
    norm = 1 / (2 * np.pi * spat_sq)

    # Loop through spatial positions
    yy, xx = np.indices(cube.shape[1:])

    for i, (y, x) in enumerate(zip(yy.ravel(), xx.ravel())):

        # model_yx = model(spec_axis, params[y, x, 0], params[y, x, 1],
        #                  params[y, x, 2],)

        # mod_resids = np.sum((model_yx - cube[:, y, x])**2)

        # If the peak SNR falls below null_model_thresh, check
        # whether the model is needed
        # if params[y, x, 0] / cube_err[y, x] <= null_model_thresh:

        model_yx = np.zeros_like(spec_axis, dtype=np.float)

        skip_reg = []

        for j in range(ncomps):
            # Transform 'on' parameter from real line to 0, 1
            # on_val = 1. / (1. + np.exp(- params[y, x, 0]))
            # # on_val = params[y, x, 0]

            # if on_val < null_model_thresh:
            #     null_resids = np.sum(cube[:, y, x]**2)
            #     resids = null_resids
            #     skip_reg = True
            #     # if null_resids < mod_resids * (3 - 1):
            #     #     resids = null_resids
            #     #     skip_reg = True
            #     # else:
            #     #     resids = mod_resids
            #     #     skip_reg = False

            # else:

            if params[y, x, 3 * j] >= null_model_thresh:

                model_yx += model(spec_axis, params[y, x, 3 * j],
                                  params[y, x, 3 * j + 1],
                                  params[y, x, 3 * j + 2],)
                skip_reg.append(False)
            else:
                skip_reg.append(True)

        # import matplotlib.pyplot as plt
        # plt.plot(cube[:, y, x])
        # plt.plot(model_yx)
        # plt.draw()
        # input("?")
        # plt.clf()

        mod_resids = np.sum(loss_func((model_yx - cube[:, y, x])**2))

        resids = mod_resids

        # If I find a good way to quickly test for the lack of a component
        # change this to a per-component True/False
        # skip_reg = [False] * ncomps

        lnlike += resids

        # Regularize vs. params
        # if use_reg and not skip_reg:
        if use_reg:
            # pars = params[y, x, :]

            # Only include regions whose distance is within the spatial scale
            # Exclude double counting neighbours by excluding pixels already
            # before y, x in yy and xx.
            all_dists = np.sqrt((yy.ravel() - y)**2 + (xx.ravel() - x)**2)
            neighbs = np.where(all_dists[i + 1:] < spatial_scale)

            for j in range(ncomps):
                if skip_reg[j]:
                    continue
                pars = params[y, x, 3 * j: 3 * j + 3]

                for y2, x2 in zip(yy.ravel()[i + 1:][neighbs],
                                  xx.ravel()[i + 1:][neighbs]):

                    # par_diff = pars - params[y2, x2, :]
                    par_diff = pars - params[y2, x2, 3 * j: 3 * j + 3]

                    dist = (x2 - x)**2 + (y2 - y)**2

                    if vel_surf is not None:
                        vel_surf_diff = \
                            np.abs(vel_surf[y, x] - vel_surf[y2, x2])
                        if np.abs(par_diff[1]) < vel_surf_diff:
                            par_diff[1] = 0.

                    dist_weight = norm * np.exp(- 0.5 * dist / spat_sq)

                    err_weight = cube_err[y, x] * cube_err[y2, x2]

                    # Since W is diagonal, could just multiply arrays with
                    # diagonal. But this is the general form for now
                    lnlike += err_weight * dist_weight * \
                        np.sqrt(sum([par_diff[k]**2 * param_weights[k] for k in
                                     range(3)]))

    return lnlike


def generate_bounds(params, vmin, vmax, amp_up=None, sigma_down=None, sigma_up=None):
    '''
    Create fit bounds for the Gaussian parameters.

    Amplitude > 0
    vmin <= v0 <= vmax
    sigma > 0
    '''

    ncomps = params.shape[-1] // 3

    low_bounds = []
    up_bounds = []

    if amp_up is None:
        amp_up = np.inf
    if sigma_up is None:
        sigma_up = np.inf
    if sigma_down is None:
        sigma_down = 0.

    # Loop through spatial positions
    yy, xx = np.indices(params.shape[:-1])

    for i, (y, x) in enumerate(zip(yy.ravel(), xx.ravel())):

        for j in range(ncomps):

            # Amp
            low_bounds.append(0.)
            up_bounds.append(amp_up)
            # Mean
            low_bounds.append(vmin)
            up_bounds.append(vmax)
            # Std
            low_bounds.append(sigma_down)
            up_bounds.append(sigma_up)

    return (low_bounds, up_bounds)
