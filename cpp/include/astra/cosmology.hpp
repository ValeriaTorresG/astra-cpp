// Cosmología plana ΛCDM con radiación y neutrinos masivos, con la misma
// convención que astropy.cosmology.FlatLambdaCDM, y paso de (RA, DEC, z) a
// coordenadas cartesianas comóviles en Mpc.
#pragma once

#include <cstddef>
#include <vector>

namespace astra {

class FlatLambdaCDM {
public:
    // m_nu en eV, una entrada por especie (astropy reparte Neff en 3 especies).
    FlatLambdaCDM(double H0, double Om0, double Tcmb0, double Neff, std::vector<double> m_nu);

    // Parámetros de astropy.cosmology.Planck18.
    static FlatLambdaCDM planck18();

    double H0() const { return H0_; }
    double Om0() const { return Om0_; }
    double Ode0() const { return Ode0_; }
    double Ogamma0() const { return Ogamma0_; }
    double Onu0() const { return Onu0_; }
    double hubble_distance() const;  // c / H0 en Mpc

    double nu_relative_density(double z) const;
    double inv_efunc(double z) const;  // 1 / E(z)

    // Distancia comóvil en Mpc por integración directa (Gauss-Legendre compuesto).
    double comoving_distance(double z) const;

private:
    double H0_, Om0_, Tcmb0_, Neff_;
    std::vector<double> nu_y_;  // m_nu / (k_B T_nu0) de las especies masivas
    int n_massless_nu_ = 0;
    double Ogamma0_ = 0, Onu0_ = 0, Ode0_ = 0;
};

// Tabla de distancia comóvil en [0, zmax] interpolada con Hermite cúbico usando
// la derivada exacta dD/dz = D_H / E(z); el error relativo es < 1e-12.
class ComovingDistanceTable {
public:
    ComovingDistanceTable(const FlatLambdaCDM& cosmo, double zmax, double dz = 1e-3);
    double operator()(double z) const;
    double zmax() const { return zmax_; }

private:
    double dz_, zmax_;
    std::vector<double> dist_, deriv_;
};

// xyz[3*i + k] = D_C(z_i) * (cos δ cos α, cos δ sin α, sin δ). Lanza si hay
// valores no finitos o z <= 0.
void sky_to_cartesian(const double* ra_deg, const double* dec_deg, const double* z,
                      std::size_t n, const FlatLambdaCDM& cosmo, double* xyz, int threads);

}  // namespace astra
