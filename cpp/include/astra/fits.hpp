// minimal fits binary table (bintable) reader without external dependencies.
//
// read numeric scalar columns (b, i, j, k, e, d) and convert them to double.
// read rows with pread in large parallel blocks; read sparse requested rows
// individually to avoid scanning the entire file.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "astra/common.hpp"

namespace astra {

struct FitsColumn {
    std::string name;
    char type = 0;           // tform code: l x b i j k a e d c m p q
    long repeat = 1;
    std::size_t offset = 0;  // bytes from the start of the row
    std::size_t width = 0;   // bytes occupied by the column in the row
};

class FitsTable {
public:
    // hdu: extension index (0 = primary, 1 = first extension).
    explicit FitsTable(std::string path, int hdu = 1);

    const std::string& path() const { return path_; }
    Index nrows() const { return nrows_; }
    std::size_t row_bytes() const { return row_bytes_; }
    std::size_t data_offset() const { return data_offset_; }
    const std::vector<FitsColumn>& columns() const { return columns_; }

    bool has_column(const std::string& name) const;
    const FitsColumn& column(const std::string& name) const;  // case-insensitive

private:
    std::string path_;
    Index nrows_ = 0;
    std::size_t row_bytes_ = 0;
    std::size_t data_offset_ = 0;
    std::vector<FitsColumn> columns_;
};

// table read request: rows in ascending order (empty + all_rows = all rows)
// written starting at out_offset.
struct FitsReadRequest {
    const FitsTable* table = nullptr;
    const Index* rows = nullptr;  // sorted, unique; ignored if all_rows
    Index n_rows = 0;
    bool all_rows = false;
    Index out_offset = 0;
};

// output column: row r's value goes in data[r * stride] (stride in elements),
// allowing direct writes into a column of an (n, 3) array.
struct OutputColumn {
    double* data = nullptr;
    std::ptrdiff_t stride = 1;
};

// read columns `names` from multiple tables and write them as doubles at row
// `out_offset + i` of each output. distribute (table, block) tasks among
// `threads` threads, allowing parallel reads even for one table.
void read_columns(const std::vector<FitsReadRequest>& requests,
                  const std::vector<std::string>& names,
                  const std::vector<OutputColumn>& outputs,
                  int threads,
                  std::size_t block_bytes = std::size_t(64) << 20);

}  // namespace astra
