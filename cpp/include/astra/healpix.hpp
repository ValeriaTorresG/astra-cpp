// healpix pixel indices in the ring scheme, following healpix_base::loc2pix
// from healpix c++ (the library behind healpy.ang2pix / healpy.vec2pix).
#pragma once

#include <cmath>
#include <cstdint>

namespace astra::healpix {

inline bool valid_nside(std::int64_t nside) { return nside >= 1 && nside <= (std::int64_t(1) << 29); }

inline std::int64_t npix(std::int64_t nside) { return 12 * nside * nside; }

namespace detail {

inline double fmodulo(double v1, double v2) {
    if (v1 >= 0) return v1 < v2 ? v1 : std::fmod(v1, v2);
    const double tmp = std::fmod(v1, v2) + v2;
    return tmp == v2 ? 0.0 : tmp;
}

// z = cos(theta), phi in radians, sth = sin(theta) (used near the poles if have_sth).
inline std::int64_t loc2pix_ring(std::int64_t nside, double z, double phi, double sth, bool have_sth) {
    constexpr double inv_halfpi = 0.6366197723675813430755350534900574;
    constexpr double twothird = 2.0 / 3.0;
    const double za = std::abs(z);
    const double tt = fmodulo(phi * inv_halfpi, 4.0);  // in [0, 4)
    if (za <= twothird) {  // equatorial region
        const std::int64_t nl4 = 4 * nside;
        const double temp1 = double(nside) * (0.5 + tt);
        const double temp2 = double(nside) * z * 0.75;
        const auto jp = std::int64_t(temp1 - temp2);  // ascending edge line
        const auto jm = std::int64_t(temp1 + temp2);  // descending edge line
        const std::int64_t ir = nside + 1 + jp - jm;  // ring number counted from z = 2/3, in {1, 2n+1}
        const std::int64_t kshift = 1 - (ir & 1);
        const std::int64_t t1 = jp + jm - nside + kshift + 1 + nl4 + nl4;
        const std::int64_t ip = (t1 >> 1) % nl4;
        const std::int64_t ncap = 2 * nside * (nside - 1);
        return ncap + (ir - 1) * nl4 + ip;
    }
    // polar caps
    const double tp = tt - double(std::int64_t(tt));
    const double tmp = (za < 0.99 || !have_sth) ? double(nside) * std::sqrt(3 * (1 - za))
                                                : double(nside) * sth / std::sqrt((1.0 + za) / 3.0);
    const auto jp = std::int64_t(tp * tmp);
    const auto jm = std::int64_t((1.0 - tp) * tmp);
    const std::int64_t ir = jp + jm + 1;  // ring number counted from the closest pole
    std::int64_t ip = std::int64_t(tt * double(ir));
    if (ip >= 4 * ir) ip = 4 * ir - 1;
    return z > 0 ? 2 * ir * (ir - 1) + ip : npix(nside) - 2 * ir * (ir + 1) + ip;
}

}  // namespace detail

// healpy.ang2pix(nside, theta, phi) (ring), angles in radians.
inline std::int64_t ang2pix_ring(std::int64_t nside, double theta, double phi) {
    return detail::loc2pix_ring(nside, std::cos(theta), phi, std::sin(theta),
                                theta < 0.01 || theta > 3.14159 - 0.01);
}

// pixel of (ra, dec) in degrees, as healpy.ang2pix(nside, radians(90 - dec), radians(ra mod 360)).
inline std::int64_t radec2pix_ring(std::int64_t nside, double ra_deg, double dec_deg) {
    constexpr double deg = 0.017453292519943295769236907684886;
    double ra = std::fmod(ra_deg, 360.0);  // same as np.mod(ra, 360)
    if (ra < 0) ra += 360.0;
    return ang2pix_ring(nside, (90.0 - dec_deg) * deg, ra * deg);
}

// healpy.vec2pix(nside, x, y, z) (ring); the vector need not be normalized.
inline std::int64_t vec2pix_ring(std::int64_t nside, double x, double y, double z) {
    const double xl = 1.0 / std::sqrt(x * x + y * y + z * z);
    const double phi = (x == 0.0 && y == 0.0) ? 0.0 : std::atan2(y, x);
    const double nz = z * xl;
    if (std::abs(nz) > 0.99) return detail::loc2pix_ring(nside, nz, phi, std::sqrt(x * x + y * y) * xl, true);
    return detail::loc2pix_ring(nside, nz, phi, 0.0, false);
}

}  // namespace astra::healpix
