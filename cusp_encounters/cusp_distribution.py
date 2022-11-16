import numpy as np
from . import power
from . import decorators
from . import peak_statistics

# Planck 18 cosmology
#cosm = {'omega_cdm': 0.26067,'omega_de': 0.69889, 'omega_baryon': 0.04897,'hubble': 0.6766,'ns': 0.9665,'sigma8': 0.8102,'tau': 0.0561,'A_s': None,'neutrino_mass': 0.06,'w0': -1,'wa': 0}
cosm_planck_18 = {'omega_cdm': 0.26067, 'omega_baryon': 0.04897, 'hubble': 0.6766, 'ns': 0.9665, 'tau': 0.0561, 'A_s': 2.105e-09, 'neutrino_mass': 0.0}
class CuspDistribution():
    def __init__(self, cachedir=None, zref=30.6, kmatch=5e3, cosmology=cosm_planck_18):
        self.aref = 1./(1.+zref)
        
        #ad =5.332e-12
        ad = 5.586938741852608e-12
        lfs = free_streaming_length(mx_gev=100., td_mev=30., ad=ad, amax=1., h=cosmology["hubble"], 
                                    omega_m=(cosmology["omega_cdm"]+cosmology["omega_baryon"]), 
                                    omega_l=1-(cosmology["omega_cdm"]+cosmology["omega_baryon"]), 
                                    Tcmb=2.7255, geff_ur=3.044)
        print("Calculated kfs = %g" % (1./lfs))
        print("Not yet using this value")
            
        self.cachedir = cachedir
        self.kfs = 1.06e6
        self.k = np.logspace(-6, 10, 1000)
        self.cosmology = cosmology
                
        def get_base_spectrum(zref, kmatch, cosmology):
            self.myclass = power.get_class(z=zref, kmax=2.*kmatch, cosm=cosmology)
            return power.dimless_cdmpower_all_scales(self.k, self.myclass, z=zref, kmatch=kmatch, A_s=cosmology["A_s"]) / self.aref**2
        
        if self.cachedir is not None:
            get_base_spectrum = decorators.h5cache(get_base_spectrum, file="%s/power_spectrum.hdf5" % self.cachedir)
            
        self.Dk_cdm_base = get_base_spectrum(zref, kmatch, self.cosmology)
        self.Dk_cdm = self.Dk_cdm_base * self.wimp_transfer_function(self.k)
        
        self.ps = peak_statistics.PeakStatistics(self.k, self.Dk_cdm)

    def wimp_transfer_function(self, k):
        return np.exp(-(k/self.kfs)**2)
    
    def sample_peaks(self, N = 100000, seed=None, numin=0.):
        def cached_sample_peaks(N, k, Dk, numin=0., seed=None):
            #peakstatistics = peak_statistics.PeakStatistics(k, Dk)
            
            if seed is not None:
                np.random.seed(seed)

            return self.ps.sample_peaks(N, numin=numin)
            
        if self.cachedir is not None:
            cached_sample_peaks = decorators.h5cache(cached_sample_peaks, file="%s/peaks.hdf5" % self.cachedir)

        return cached_sample_peaks(N, k=self.k, Dk=self.Dk_cdm, numin=numin, seed=seed)
    
    def sample_cusps(self, N = 100000, seed=None, numin=0., units_pc=True):
        nu,x,e,p = self.sample_peaks(N, seed=seed, numin=numin)
        
        G = 43.0071057317063e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
        rho0 = 3.*(100.*self.cosmology["hubble"])**2 / (8.*np.pi*G) * self.cosmology["omega_cdm"] # Msol/Mpc

        fec = np.zeros_like(e)
        valid = (e**2 - p*np.abs(p))  < 0.26
        fec[valid] = ellipsoidal_collapse_correction(e[valid], p[valid], maxiter=40)

        deltaref = nu*self.ps.sigma0 * self.aref  
        # Note that we have normalized everything so that delta(aref) = delta * aref
        # where aref is during matter domination
        g = 0.901
        acoll = (fec * 1.686 / deltaref)**(1./g) * self.aref 
        acoll[acoll == 0.] = np.nan

        delta = nu * self.ps.sigma0
        L = -x * self.ps.sigma2

        R = np.sqrt(np.abs(delta/L))  #/ h

        A = 24. * rho0 * acoll**(-1.5) * R**1.5
        rcusp = 0.11 * acoll * R
        Mcusp = (8.*np.pi/3.) * A * rcusp**1.5

        def J_of_cusp(A, rcusp, rcore):
            return (4.*np.pi / 3.) * A**2 * (1. + 3.*np.log(rcusp/rcore))

        def rcore_of_cusp(A, fmax):
            """Uses units of Msol and Mpc"""
            return (2.89896590246533e-5 * G**-3 * fmax**-2 / A)**(2./9.)
        print("Warning, I have to insert the correct WIMP fmax for m!=100mev here")
        rcore = rcore_of_cusp(A, fmax_wimp())

        J = J_of_cusp(A, rcusp, rcore)

        if units_pc:
            res = dict(A=A/1e9, R=R*1e6, rcusp=rcusp*1e6, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J/1e18, rcore=rcore*1e6)
        else:
            res = dict(A=A, R=R, rcusp=rcusp, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J, rcore=rcore)

        return res

def omega_rad(a, Tcmb = 2.7255, h=0.678, geff_ur=3.044):
    c = 299792458.0
    sig = 5.670374419e-8 # Stefan boltzmann constnat

    rho_r = 4./c * sig * Tcmb**4

    H0 = 100. * h
    rho_crit = 3.*(H0*1e3 / (3.085677581491367e16*1e6))**2 / (8.*np.pi*6.6743e-11)

    omega_photon = (rho_r)/(rho_crit*c**2)
    
    Tnu = Tcmb * (4./11.)**(1./3.)
    
    omega_nu = 7./22. * geff_ur * (4./11.)**(1./3.) * omega_photon
    
    return (omega_nu + omega_photon) * a**-4
    
    #return (geff_ur/2.) * omega_photon * a**-4 

def hubble(a, h=0.678, omega_m=0.3, omega_l=0.7, Tcmb=2.7255, geff_ur=3.044):
    return 100.*h * np.sqrt(omega_m*a**-3 + omega_l + omega_rad(a, h=h, Tcmb=Tcmb, geff_ur=geff_ur))


def free_streaming_length(mx_gev=100., td_mev=30., ad=5.332e-12, amax=1., h=0.678, omega_m=0.3, omega_l=0.7, cumulative=False, Tcmb=2.7255, geff_ur=3.044):
    """https://arxiv.org/pdf/astro-ph/0607319.pdf eqn.48"""
    def f(a):
        Hl = hubble(a, omega_m=omega_m, omega_l=omega_l, Tcmb=Tcmb, geff_ur=geff_ur, h=h) / 299792458.0e-3 # Hubble in 1/Mpc

        return 1./(Hl *a**3)
    
    mx_ev, Td_ev = mx_gev*1e9, td_mev*1e6
    
    # At early times the conformal time is propotional to a
    # therefure a/a_d = tau/tau_d
    astart = ad*1.05  

    a = np.logspace(np.log10(astart),np.log10(amax), 100000)
    
    if cumulative:
        integral = peak_statistics.cum_simpson(f, a)
    
        return a, ad * np.sqrt(6.*Td_ev/5./mx_ev) * integral
    else:
        from scipy.integrate import simps
        integral = simps(f(a), a)
    
        return ad * np.sqrt(6.*Td_ev/5./mx_ev) * integral

def ellipsoidal_collapse_correction(e, p, maxiter=20):
    def myfunc(f, e, p):
        return 1+0.47*(5*(e**2-p*np.abs(p))*f**2)**0.615
    
    fnew = np.ones_like(e) * 1.1
    for i in range(0,20):
        fnew = myfunc(fnew, e, p)
        
    return fnew


def fmax_wimp(h=0.68, omega_dm=0.26,  mx=100, Td=30., ad=5.332e-12):
    """Following https://arxiv.org/pdf/2207.05082.pdf

    mx : WIMP mass in GeV
    Td : Decoupling Temperature in MeV
    omgega_dm: dark matter (not full matter) density paramater, mx in kev
    ad : scale factor of decoupling (where the temperature of the universe is Td)
         This can be put to None to use an approximation by the neutrino temperature
         which may have errors of order 10% if the wimp decoupled a bit before the
         neutrinos

    the phase space density of WIMP's
    """
    mev, Tdev = mx*1e9, Td*1e6
    
    c = 299792458.0
    G = 43.0071057317063 * 1e-10

    if ad is None:
        print("Approximating ad by assuming evaluating T(a_d)=Td while using the temperature T(a) of the Neutrino background.\n"
              "This may give inaccurate results by 10-20%. For full accuracy use a full thermal history and determine ad")
        Tcmb = 2.725 #K
        kb = 8.617333262e-5 # eV/kelvin
        Tnu = Tcmb*(4./11.)**(1./3.) * kb   # in eV
        ad = (Tnu/Tdev)

    v0 =  np.sqrt(Tdev * mev)*ad / mev * c / 1e3  # velocity today in km/s

    rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*G) * omega_dm

    return (2.*np.pi)**(-3./2.) * v0**-3 * rho_dm

def peaks_to_cusps(nu, x, e, p, units_pc=False):
    h = cosm.pars["hubble"]
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    H0 = 100.*h
    rho0 = 3.*H0**2 / (8.*np.pi*G) * cosm.pars["omega_cdm"] # Msol/Mpc
    
    fec = np.zeros_like(e)
    valid = (e**2 - p*np.abs(p))  < 0.26
    fec[valid] = ellipsoidal_collapse_correction(e[valid], p[valid], maxiter=40)

    aref = 1./(1.+zref)
    #deltaref = nu*c.sigma0*cosm.get_growth_z(aref)
    deltaref = nu*c.sigma0 * aref  
    # Note that we have normalized everything so that delta(aref) = delta * aref
    # where aref is during matter domination
    g = 0.901
    acoll = (fec * 1.686 / deltaref)**(1./g) * aref # This cannot be correct yet!
    #acoll = (fec * 1.686 / deltaref)*aref#**(1./g) * aref # This cannot be correct yet!


    #acoll = fec*1.686/(nu*c.sigma0)
    delta = nu * c.sigma0
    L = -x * c.sigma2

    R = np.sqrt(np.abs(delta/L))  #/ h

    A = 24. * rho0 * acoll**(-1.5) * R**1.5
    rcusp = 0.11 * acoll * R
    Mcusp = (8.*np.pi/3.) * A * rcusp**1.5
    
    def J_of_cusp(A, rcusp, rcore):
        return (4.*np.pi / 3.) * A**2 * (1. + 3.*np.log(rcusp/rcore))
    
    def rcore_of_cusp(A, fmax):
        """Uses units of Msol and Mpc"""
        G = 43.0071057317063 * 1e-10
        #Arc45 = 2.89896590246533e-5 * G**-3 * fmax**-2
        return (2.89896590246533e-5 * G**-3 * fmax**-2 / A)**(2./9.)
    rcore = rcore_of_cusp(A, fmax_wimp())
    
    J = J_of_cusp(A, rcusp, rcore)
    
    if units_pc:
        res = dict(A=A/1e9, R=R*1e6, rcusp=rcusp*1e6, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J/1e18, rcore=rcore*1e6)
    else:
        res = dict(A=A, R=R, rcusp=rcusp, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J, rcore=rcore)
    
    return res

