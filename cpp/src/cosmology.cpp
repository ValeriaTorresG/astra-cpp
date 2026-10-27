#include "astra/cosmology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "astra/common.hpp"

namespace astra {
namespace {

// Constantes CODATA 2018 / IAU 2015 (las mismas de astropy.constants).
constexpr double kC = 299792458.0;               // m / s
constexpr double kG = 6.67430e-11;               // m^3 / (kg s^2)
constexpr double kSigmaSB = 5.670374419e-8;      // W / (m^2 K^4)
constexpr double kBoltzmannEV = 8.617333262e-5;  // eV / K
constexpr double kMpc = 3.0856775814913673e22;   // m
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// Gauss-Legendre de 8 nodos en [-1, 1].
constexpr double kGLx[4] = {0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
constexpr double kGLw[4] = {0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};

template <class F>
double gauss_legendre(F&& f, double a, double b) {
    const double h = 0.5 * (b - a), m = 0.5 * (a + b);
    double s = 0.0;
    for (int i = 0; i < 4; ++i) s += kGLw[i] * (f(m - h * kGLx[i]) + f(m + h * kGLx[i]));
    return s * h;
}

}  // namespace

FlatLambdaCDM::FlatLambdaCDM(double H0, double Om0, double Tcmb0, double Neff, std::vector<double> m_nu)
    : H0_(H0), Om0_(Om0), Tcmb0_(Tcmb0), Neff_(Neff) {
    // Igual que astropy.cosmology.FLRW: densidad de fotones y neutrinos.
    const double h0_si = H0 * 1e3 / kMpc;
    const double rho_crit = 3.0 * h0_si * h0_si / (8.0 * kPi * kG);
    const double a_rad = 4.0 * kSigmaSB / (kC * kC * kC);
    Ogamma0_ = a_rad * std::pow(Tcmb0, 4) / rho_crit;
    const double tnu0 = 0.7137658555036082 * Tcmb0;  // (4/11)^(1/3) Tcmb0
    for (double m : m_nu) {
        if (m > 0) nu_y_.push_back(m / (kBoltzmannEV * tnu0));
        else ++n_massless_nu_;
    }
    Onu0_ = Ogamma0_ * nu_relative_density(0.0);
    Ode0_ = 1.0 - Om0_ - Ogamma0_ - Onu0_;
}

FlatLambdaCDM FlatLambdaCDM::planck18() {
    return FlatLambdaCDM(67.66, 0.30966, 2.7255, 3.046, {0.0, 0.0, 0.06});
}

double FlatLambdaCDM::hubble_distance() const { return kC * 1e-3 / H0_; }

double FlatLambdaCDM::nu_relative_density(double z) const {
    // Ajuste de Komatsu et al. (2011), como en astropy.
    constexpr double prefac = 0.22710731766;  // 7/8 (4/11)^(4/3)
    if (nu_y_.empty()) return prefac * Neff_;
    constexpr double p = 1.83, invp = 0.54644808743, k = 0.3173;
    double rel_mass = n_massless_nu_;
    for (double y : nu_y_) rel_mass += std::pow(1.0 + std::pow(k * y / (1.0 + z), p), invp);
    const double neff_per_nu = Neff_ / double(nu_y_.size() + n_massless_nu_);
    return prefac * neff_per_nu * rel_mass;
}

double FlatLambdaCDM::inv_efunc(double z) const {
    const double zp1 = 1.0 + z;
    const double Or = Ogamma0_ * (1.0 + nu_relative_density(z));
    return 1.0 / std::sqrt(zp1 * zp1 * zp1 * (Or * zp1 + Om0_) + Ode0_);
}

double FlatLambdaCDM::comoving_distance(double z) const {
    const int steps = std::max(1, int(std::ceil(std::fabs(z) / 1e-3)));
    const double h = z / steps;
    double s = 0.0;
    for (int i = 0; i < steps; ++i)
        s += gauss_legendre([this](double x) { return inv_efunc(x); }, i * h, (i + 1) * h);
    return hubble_distance() * s;
}

ComovingDistanceTable::ComovingDistanceTable(const FlatLambdaCDM& cosmo, double zmax, double dz) : dz_(dz) {
    if (!(zmax > 0) || !(dz > 0)) throw std::invalid_argument("zmax y dz deben ser positivos");
    const std::size_t n = std::size_t(std::ceil(zmax / dz)) + 1;
    zmax_ = dz * double(n - 1);
    dist_.resize(n);
    deriv_.resize(n);
    const double dh = cosmo.hubble_distance();
    auto f = [&cosmo](double x) { return cosmo.inv_efunc(x); };
    dist_[0] = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) dist_[i] = dist_[i - 1] + dh * gauss_legendre(f, (i - 1) * dz, i * dz);
        deriv_[i] = dh * cosmo.inv_efunc(i * dz);
    }
}

double ComovingDistanceTable::operator()(double z) const {
    if (!(z >= 0.0 && z <= zmax_)) throw std::out_of_range("z fuera de la tabla de distancias");
    const double u = z / dz_;
    const std::size_t i = std::min(std::size_t(u), dist_.size() - 2);
    const double t = u - double(i), t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * dist_[i] + (t3 - 2 * t2 + t) * dz_ * deriv_[i] +
           (-2 * t3 + 3 * t2) * dist_[i + 1] + (t3 - t2) * dz_ * deriv_[i + 1];
}

void sky_to_cartesian(const double* ra_deg, const double* dec_deg, const double* z, std::size_t n,
                      const FlatLambdaCDM& cosmo, double* xyz, int threads) {
    const int nt = resolve_threads(threads);
    double zmax = 0.0;
    std::size_t bad = 0;
#pragma omp parallel for num_threads(nt) reduction(max : zmax) reduction(+ : bad) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ra_deg[i]) || !std::isfinite(dec_deg[i]) || !std::isfinite(z[i]) || z[i] <= 0) ++bad;
        else zmax = std::max(zmax, z[i]);
    }
    if (bad) throw std::runtime_error(std::to_string(bad) + " filas con RA/DEC no finitos o z <= 0");
    if (n == 0) return;
    const ComovingDistanceTable table(cosmo, zmax * 1.001 + 1e-3);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const double d = table(z[i]);
        const double ra = ra_deg[i] * kDeg, dec = dec_deg[i] * kDeg;
        const double cd = std::cos(dec);
        xyz[3 * i + 0] = d * cd * std::cos(ra);
        xyz[3 * i + 1] = d * cd * std::sin(ra);
        xyz[3 * i + 2] = d * std::sin(dec);
    }
}

}  // namespace astra
