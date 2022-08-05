import lalsimulation as lalsim
import lal
import numpy as np
import matplotlib as pl
import seaborn as sns
import glob
import scipy.stats as st
import os
import emcee

def plot_radii_gaussian_kde(datafile, label):
    # Plot the meshgrid & scatter of a mass-radius distribution

    pl.clf()
    #pl.rcParams.update({'font.size': 18})
    #pl.figure(figsize=(15, 10))

    data = np.loadtxt(datafile)
    m = data[:,0] 
    r = data[:,1] / 1000

    m_min, m_max = min(m), max(m)
    r_min, r_max = min(r), max(r)

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arrays
    positions = np.vstack([mm.ravel(), rr.ravel()])
    pairs = np.vstack([m, r])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, mm.shape)

    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(r_min, r_max)

    ax.pcolormesh(mm, rr, f)
    ax.set_xlabel('Mass')
    ax.set_ylabel('Radius (km)')
    pl.scatter(m,r,s=1,color="black")
    pl.title("Mass-Radius Distribution")
    pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(label), bbox_inches='tight') # label="APR4_EPP_m(m_sigma)_r(r_sigma)_kde_mesh_scatter"

class sampler:

    def __init__(self):

        # parameter space bounds
        self.p1_l_b, self.p1_u_b = 32.973, 33.888
        self.g1_l_b, self.g1_u_b = 2.216, 4.070
        self.g2_l_b, self.g2_u_b = 1.472, 3.791
        self.g3_l_b, self.g3_u_b = 1.803, 3.660

    def obtain_kernel(self, filename):

        masses, radii = np.loadtxt(filename, unpack=True) # Mass Radius distribution of mock data
        pairs = np.vstack([masses, radii])
        self.kernel = st.gaussian_kde(pairs)

    def log_posterior(self, parameters):

        p1, g1, g2, g3 = parameters

        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def log_likelihood(self, params, kernel):
        # Finds integral of eos curve over the kde of a mass-radius posterior sample

        p1, g1, g2, g3 = params

        # MCMC code travels outside the parameter space sometimes; if statement is
        # needed to make sure eos pointer isn't made (resulting in a seg fault)
        if ((self.p1_l_b <= p1 <= self.p1_u_b) & (self.g1_l_b <= g1 <= self.g1_u_b) &
            (self.g2_l_b <= g2 <= self.g2_u_b) & (self.g3_l_b <= g3 <= self.g3_u_b)):
            try: 
                eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
                fam = lalsim.CreateSimNeutronStarFamily(eos)
                m_min = 1.0
                max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
                max_mass = int(max_mass*1000)/1000
                m_grid = np.linspace(m_min, max_mass, 1000)
                m_grid = m_grid[m_grid <= max_mass]

                working_masses = []
                working_radii = []
                for m in m_grid:
                    try:
                        r = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                        working_masses.append(m)
                        working_radii.append(r)
                    except RuntimeError:
                        continue

                return np.log(np.sum(np.array(kernel(np.vstack([working_masses, working_radii])))*np.diff(working_masses)[0]))

            except RuntimeError: return - np.inf
            except IndexError: return - np.inf
        else:
            return - np.inf

    def log_prior(self, parameters):

        p1, g1, g2, g3 = parameters

        # If sample values are within bounds, return 0, Else return - infinite
        if ((self.p1_l_b <= p1 <= self.p1_u_b) & (self.g1_l_b <= g1 <= self.g1_u_b) &
            (self.g2_l_b <= g2 <= self.g2_u_b) & (self.g3_l_b <= g3 <= self.g3_u_b)):
            return 0
        else:
            return - np.inf

    # randomly selected walker starting points
    def n_walker_points(self, walkers):

        p1_incr, g1_incr, g2_incr, g3_incr = .4575, .927, 1.1595, .9285
        log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 # defaults

        points = []
        for walker in range(walkers):

            p1_choice = ((log_p1_SI - p1_incr) +
                        ((2 * p1_incr) * np.random.random()))
            g1_choice = ((g1 - g1_incr) +
                        ((2 * g1_incr) * np.random.random()))
            g2_choice = ((g2 - g2_incr) +
                        ((2 * g2_incr) * np.random.random()))
            g3_choice = ((g3 - g3_incr) +
                        ((2 * g3_incr) * np.random.random()))

            points.append([p1_choice,g1_choice,g2_choice,g3_choice])

        return(np.array(points))

    def run_mcmc(self, label, sample_size=5000):

        ndim, nwalkers = 4, 10
        p0 = self.n_walker_points(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
        sampler.run_mcmc(p0, sample_size)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        outputfile = "emcee_files/runs/{}.txt".format(label)
        np.savetxt(outputfile, flat_samples)

def p_vs_rho(filename, label, N, plot=True):

    samples = np.loadtxt(filename)

    max_log_pressures = []
    for sample in samples: # Obtain max pressure for each sample

        p1, g1, g2, g3 = sample
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        max_log_pressures.append(max_log_pressure)

    global_max_log_pressure = max(max_log_pressures) # max maximum pressure

    min_log_pressure = 32.0
    logp_grid = np.linspace(min_log_pressure, global_max_log_pressure, N)

    density_matrix = []
    for lp in logp_grid:

        density_grid = []
        for sample in samples:

            p1, g1, g2, g3 = sample
            eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
            density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)

        density_matrix.append(density_grid)

    lower_bound = []
    median = []
    upper_bound = []
    trouble_p_vals = []
    counter = 0
    for p_rhos in density_matrix:

        try:
            bins, bin_bounds = np.histogram(p_rhos,bins=50,density=True)
            counter += 1
        except IndexError: # Meant to catch error in density (low pressure) region. Doesn't apply anymore
            trouble_p_vals.append(logp_grid[counter])
            counter += 1
            continue

        bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
        order = np.argsort(-bins)
        bins_ordered = bins[order]
        bin_cent_ord = bin_centers[order]
        include = np.cumsum(bins_ordered) < 0.9 * np.sum(bins)
        include[np.sum(include)] = True
        lower_bound.append(min(bin_cent_ord[include]))
        median.append(np.median(p_rhos))
        upper_bound.append(max(bin_cent_ord[include]))

    logp_grid = logp_grid[~np.isin(logp_grid,trouble_p_vals)]
    rho_vals = [logp_grid, lower_bound, median, upper_bound]
    outputfile = "emcee_files/runs/p_vs_rho_{}.txt".format(label)
    np.savetxt(outputfile, rho_vals)

    if plot:

        pl.clf()
        min_log_pressure = 32.0
        eos = lalsim.SimNeutronStarEOSByName("APR4_EPP")
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        logp_grid = np.linspace(min_log_pressure, max_log_pressure, N)

        density_grid = []
        for lp in logp_grid:

            density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)

        ax = pl.gca()
        ax.set_xscale("log")

        size = 1
        pl.plot(lower_bound, logp_grid, color="blue")
        pl.plot(upper_bound, logp_grid, color="blue")
        ax.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color="blue", alpha=0.5)
        pl.plot(median, logp_grid, "k--")
        #pl.plot(density_grid, logp_grid, "r-", label="APR4_EPP")

        pl.xlim([10**17, 10**19])
        pl.xlabel("Density")
        pl.ylabel("Log Pressure")
        pl.title("Pressure vs Density")
        pl.legend()
        pl.savefig("emcee_files/plots/p_vs_rho_{}.png".format(label), bbox_inches='tight')

