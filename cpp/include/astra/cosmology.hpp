// flat λcdm cosmology with radiation and massive neutrinos, using the same
// convention as astropy.cosmology.flatlambdacdm, and conversion from (ra, dec, z)
// to comoving cartesian coordinates in mpc.
#pragma once

#include <cstddef>
#include <vector>

namespace astra {

class FlatLambdaCDM {
public:
    // m_nu in ev, one entry per species (astropy distributes neff across 3 species).
    FlatLambdaCDM(double H0, double Om0, double Tcmb0, double Neff, std::vector<double> m_nu);

    // astropy.cosmology.planck18 parameters.
    static FlatLambdaCDM planck18();

    double H0() const { return H0_; }
    double Om0() const { return Om0_; }
    double Ode0() const { return Ode0_; }
    double Ogamma0() const { return Ogamma0_; }
    double Onu0() const { return Onu0_; }
    double hubble_distance() const;  // c / H0 en Mpc

    double nu_relative_density(double z) const;
    double inv_efunc(double z) const;  // 1 / E(z)

    // comoving distance in mpc by direct integration (composite gauss-legendre).
    double comoving_distance(double z) const;

private:
    double H0_, Om0_, Tcmb0_, Neff_;
    std::vector<double> nu_y_;  // m_nu / (k_B T_nu0) for massive species
    int n_massless_nu_ = 0;
    double Ogamma0_ = 0, Onu0_ = 0, Ode0_ = 0;
};

// comoving distance table in [0, zmax], interpolated with cubic hermite using
// the exact derivative dd/dz = d_h / e(z); relative error is < 1e-12.
class ComovingDistanceTable {
public:
    ComovingDistanceTable(const FlatLambdaCDM& cosmo, double zmax, double dz = 1e-3);
    double operator()(double z) const;
    double zmax() const { return zmax_; }

private:
    double dz_, zmax_;
    std::vector<double> dist_, deriv_;
};

// xyz[3*i + k] = d_c(z_i) * (cos δ cos α, cos δ sin α, sin δ). throw if
// values are nonfinite or z <= 0.
void sky_to_cartesian(const double* ra_deg, const double* dec_deg, const double* z,
                      std::size_t n, const FlatLambdaCDM& cosmo, double* xyz, int threads);

}  // namespace astra
