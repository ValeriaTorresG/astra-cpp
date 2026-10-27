// Lector mínimo de tablas binarias FITS (BINTABLE) sin dependencias externas.
//
// Lee columnas escalares numéricas (B, I, J, K, E, D) y las convierte a double.
// Las filas se leen con pread en bloques grandes y en paralelo; si las filas
// pedidas están dispersas se leen una a una para no recorrer el archivo entero.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "astra/common.hpp"

namespace astra {

struct FitsColumn {
    std::string name;
    char type = 0;           // código TFORM: L X B I J K A E D C M P Q
    long repeat = 1;
    std::size_t offset = 0;  // bytes desde el inicio de la fila
    std::size_t width = 0;   // bytes que ocupa la columna en la fila
};

class FitsTable {
public:
    // hdu: índice de la extensión (0 = primaria, 1 = primera extensión).
    explicit FitsTable(std::string path, int hdu = 1);

    const std::string& path() const { return path_; }
    Index nrows() const { return nrows_; }
    std::size_t row_bytes() const { return row_bytes_; }
    std::size_t data_offset() const { return data_offset_; }
    const std::vector<FitsColumn>& columns() const { return columns_; }

    bool has_column(const std::string& name) const;
    const FitsColumn& column(const std::string& name) const;  // sin distinguir mayúsculas

private:
    std::string path_;
    Index nrows_ = 0;
    std::size_t row_bytes_ = 0;
    std::size_t data_offset_ = 0;
    std::vector<FitsColumn> columns_;
};

// Petición de lectura sobre una tabla: filas ordenadas de forma creciente
// (vacío + all_rows = todas las filas) que se escriben a partir de out_offset.
struct FitsReadRequest {
    const FitsTable* table = nullptr;
    const Index* rows = nullptr;  // ordenadas, sin repetir; ignorado si all_rows
    Index n_rows = 0;
    bool all_rows = false;
    Index out_offset = 0;
};

// Columna de salida: el valor de la fila r va en data[r * stride] (stride en
// elementos), lo que permite escribir directo en una columna de un arreglo (N, 3).
struct OutputColumn {
    double* data = nullptr;
    std::ptrdiff_t stride = 1;
};

// Lee las columnas `names` de varias tablas y las escribe como double en la fila
// `out_offset + i` de cada salida. Las tareas (tabla, bloque) se reparten entre
// `threads` hilos, de modo que también una sola tabla se lee en paralelo.
void read_columns(const std::vector<FitsReadRequest>& requests,
                  const std::vector<std::string>& names,
                  const std::vector<OutputColumn>& outputs,
                  int threads,
                  std::size_t block_bytes = std::size_t(64) << 20);

}  // namespace astra
