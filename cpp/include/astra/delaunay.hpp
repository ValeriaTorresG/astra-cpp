// Triangulación de Delaunay 3D con CGAL y conteo de vecinos de ASTRA.
//
// Cada vértice guarda el índice original de su punto, de modo que los conteos
// NDATA/NRAND salen alineados con el arreglo de entrada sin materializar
// aristas. Los tipos de CGAL quedan ocultos tras un puntero opaco (pimpl).
//
// Modo periódico (cajas cúbicas): se agregan copias fantasma de los puntos a
// menos de `ghost_margin` de cada cara y los vecinos fantasma se cuentan como
// su punto original. Al contar se verifica que la estrella de cada punto sea
// la de la triangulación periódica exacta (todas sus esferas circunscritas
// caben en la caja ampliada); los que no pasan se reportan en `n_unverified`.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "astra/common.hpp"

namespace astra {

struct DelaunayOptions {
    int threads = 0;          // <= 1: inserción secuencial; > 1: paralela (requiere TBB)
    // Resolución por eje de la grilla de locks (modo paralelo). <= 0: automática,
    // proporcional a N^(1/3) para que cada celda de lock cubra un número similar de
    // puntos; con celdas grandes la contención entre hilos colapsa la inserción.
    int lock_grid_cells = 0;
    bool periodic = false;    // caja periódica [box_lo, box_hi)
    std::array<double, 3> box_lo{}, box_hi{};
    double ghost_margin = 0;  // ancho de la capa de copias fantasma (modo periódico)
};

// Resolución automática de la grilla de locks para n puntos.
int auto_lock_grid_cells(Index n);

struct NeighborCounts {
    std::vector<std::int32_t> ndata;
    std::vector<std::int32_t> nrand;
    Index n_unverified = 0;   // modo periódico: puntos cuya estrella no se pudo verificar
};

class Delaunay3 {
public:
    // xyz: n puntos contiguos (x, y, z) en double. Solo se lee durante la
    // construcción (CGAL guarda su propia copia de cada punto).
    Delaunay3(const double* xyz, Index n, const DelaunayOptions& options);
    ~Delaunay3();
    Delaunay3(const Delaunay3&) = delete;
    Delaunay3& operator=(const Delaunay3&) = delete;

    Index n_points() const;
    Index n_ghosts() const;
    int lock_grid_cells() const;  // 0 si la inserción fue secuencial
    bool parallel() const;
    std::size_t number_of_vertices() const;
    std::size_t number_of_finite_cells() const;
    std::size_t number_of_finite_edges() const;
    bool is_valid() const;

    // Puntos que coinciden exactamente con otro ya insertado: (punto, vértice canónico).
    // Heredan los conteos del punto canónico.
    const std::vector<std::array<Index, 2>>& duplicates() const;

    // NDATA y NRAND por punto: vecinos de Delaunay finitos que son data / random.
    NeighborCounts neighbor_counts(const std::uint8_t* is_data, int threads) const;

    // Aristas (i, j) con i < j en índices originales, únicas y ordenadas, aplanadas
    // (para validación en muestras pequeñas). En modo periódico incluye las
    // aristas que cruzan la caja.
    std::vector<Index> edges() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace astra
