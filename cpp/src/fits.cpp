#include "astra/fits.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace astra {
namespace {

constexpr std::size_t kBlock = 2880;  // fits record unit
constexpr std::size_t kCard = 80;
constexpr std::size_t kMaxGap = std::size_t(64) << 10;  // maximum gap between requested rows for reading the full span

std::string upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(' ');
    if (b == std::string::npos) return {};
    return s.substr(b, s.find_last_not_of(' ') - b + 1);
}

// value of a "key     = value / comment" card; strings have no quotes.
std::string card_value(const std::string& card) {
    std::string v = card.substr(10);
    if (const auto q = v.find('\''); q != std::string::npos) {
        const auto e = v.find('\'', q + 1);
        return trim(v.substr(q + 1, e == std::string::npos ? std::string::npos : e - q - 1));
    }
    return trim(v.substr(0, v.find('/')));
}

class File {
public:
    explicit File(const std::string& path) : fd_(::open(path.c_str(), O_RDONLY)) {
        if (fd_ < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }
    ~File() { if (fd_ >= 0) ::close(fd_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    void read_at(char* buf, std::size_t nbytes, std::size_t offset) const {
        while (nbytes > 0) {
            const ssize_t got = ::pread(fd_, buf, nbytes, static_cast<off_t>(offset));
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("read error: ") + std::strerror(errno));
            }
            if (got == 0) throw std::runtime_error("unexpected end of file");
            buf += got;
            offset += static_cast<std::size_t>(got);
            nbytes -= static_cast<std::size_t>(got);
        }
    }

private:
    int fd_;
};

// read the cards in a header starting at `offset`; return the data offset
// (aligned to 2880) and populate `keys`.
std::size_t read_header(const File& f, std::size_t offset, std::map<std::string, std::string>& keys) {
    std::vector<char> block(kBlock);
    for (;;) {
        f.read_at(block.data(), kBlock, offset);
        offset += kBlock;
        for (std::size_t c = 0; c < kBlock; c += kCard) {
            const std::string card(block.data() + c, kCard);
            const std::string key = trim(card.substr(0, 8));
            if (key == "END") return offset;
            if (card.size() > 9 && card[8] == '=') keys[key] = card_value(card);
        }
    }
}

long key_long(const std::map<std::string, std::string>& keys, const std::string& k, long def = -1) {
    const auto it = keys.find(k);
    if (it == keys.end()) {
        if (def >= 0) return def;
        throw std::runtime_error("missing FITS keyword " + k);
    }
    return std::stol(it->second);
}

std::size_t type_width(char t, long repeat) {
    switch (t) {
        case 'L': case 'B': case 'A': return std::size_t(repeat);
        case 'X': return std::size_t((repeat + 7) / 8);
        case 'I': return 2 * std::size_t(repeat);
        case 'J': case 'E': return 4 * std::size_t(repeat);
        case 'K': case 'D': case 'C': case 'P': return 8 * std::size_t(repeat);
        case 'M': case 'Q': return 16 * std::size_t(repeat);
        default: throw std::runtime_error(std::string("unsupported TFORM: ") + t);
    }
}

// scalar big-endian value -> double.
inline double decode(const char* p, char type) {
    switch (type) {
        case 'B': return static_cast<double>(static_cast<std::uint8_t>(*p));
        case 'I': { std::uint16_t u; std::memcpy(&u, p, 2); u = __builtin_bswap16(u);
                    return static_cast<double>(static_cast<std::int16_t>(u)); }
        case 'J': { std::uint32_t u; std::memcpy(&u, p, 4); u = __builtin_bswap32(u);
                    return static_cast<double>(static_cast<std::int32_t>(u)); }
        case 'K': { std::uint64_t u; std::memcpy(&u, p, 8); u = __builtin_bswap64(u);
                    return static_cast<double>(static_cast<std::int64_t>(u)); }
        case 'E': { std::uint32_t u; std::memcpy(&u, p, 4); u = __builtin_bswap32(u);
                    float v; std::memcpy(&v, &u, 4); return static_cast<double>(v); }
        case 'D': { std::uint64_t u; std::memcpy(&u, p, 8); u = __builtin_bswap64(u);
                    double v; std::memcpy(&v, &u, 8); return v; }
        default: return 0.0;
    }
}

// work unit: selected rows [begin, end) of one request, all within a
// ~block_bytes span of the file.
struct Task {
    std::size_t request;
    Index begin, end;
};

inline Index row_at(const FitsReadRequest& r, Index k) { return r.all_rows ? k : r.rows[k]; }

}  // namespace

FitsTable::FitsTable(std::string path, int hdu) : path_(std::move(path)) {
    File f(path_);
    std::size_t offset = 0;
    std::map<std::string, std::string> keys;
    for (int h = 0;; ++h) {
        keys.clear();
        const std::size_t data = read_header(f, offset, keys);
        if (h == hdu) { data_offset_ = data; break; }
        // skip this hdu's data: |bitpix|/8 * gcount * (pcount + prod(naxisi))
        const long naxis = key_long(keys, "NAXIS");
        std::size_t size = 0;
        if (naxis > 0) {
            size = 1;
            for (long i = 1; i <= naxis; ++i) size *= std::size_t(key_long(keys, "NAXIS" + std::to_string(i)));
        }
        size = std::size_t(std::labs(key_long(keys, "BITPIX")) / 8) * std::size_t(key_long(keys, "GCOUNT", 1)) *
               (std::size_t(key_long(keys, "PCOUNT", 0)) + size);
        offset = data + (size + kBlock - 1) / kBlock * kBlock;
    }
    if (keys["XTENSION"] != "BINTABLE")
        throw std::runtime_error(path_ + ": HDU " + std::to_string(hdu) + " is not BINTABLE");
    row_bytes_ = std::size_t(key_long(keys, "NAXIS1"));
    nrows_ = key_long(keys, "NAXIS2");
    const long nfields = key_long(keys, "TFIELDS");
    std::size_t col_offset = 0;
    for (long i = 1; i <= nfields; ++i) {
        const std::string tform = keys.at("TFORM" + std::to_string(i));
        std::size_t pos = 0;
        while (pos < tform.size() && std::isdigit(static_cast<unsigned char>(tform[pos]))) ++pos;
        FitsColumn c;
        c.repeat = pos ? std::stol(tform.substr(0, pos)) : 1;
        c.type = tform.at(pos);
        auto it = keys.find("TTYPE" + std::to_string(i));
        c.name = it != keys.end() ? it->second : "COL" + std::to_string(i);
        c.offset = col_offset;
        c.width = type_width(c.type, c.repeat);
        col_offset += c.width;
        columns_.push_back(std::move(c));
    }
    if (col_offset != row_bytes_)
        throw std::runtime_error(path_ + ": TFORM widths do not sum to NAXIS1");
}

bool FitsTable::has_column(const std::string& name) const {
    const std::string u = upper(name);
    return std::any_of(columns_.begin(), columns_.end(), [&](const FitsColumn& c) { return upper(c.name) == u; });
}

const FitsColumn& FitsTable::column(const std::string& name) const {
    const std::string u = upper(name);
    for (const auto& c : columns_)
        if (upper(c.name) == u) return c;
    throw std::runtime_error(path_ + ": column does not exist: " + name);
}

void read_columns(const std::vector<FitsReadRequest>& requests, const std::vector<std::string>& names,
                  const std::vector<OutputColumn>& outputs, int threads, std::size_t block_bytes) {
    if (names.size() != outputs.size()) throw std::invalid_argument("names and outputs must have the same size");

    // columns resolved for each request and validated.
    std::vector<std::vector<const FitsColumn*>> cols(requests.size());
    for (std::size_t r = 0; r < requests.size(); ++r) {
        const auto& req = requests[r];
        const FitsTable& t = *req.table;
        for (const auto& n : names) {
            const FitsColumn& c = t.column(n);
            if (c.repeat != 1 || std::strchr("BIJKED", c.type) == nullptr)
                throw std::runtime_error(t.path() + ": column " + n + " is not a numeric scalar");
            cols[r].push_back(&c);
        }
        const Index count = req.all_rows ? t.nrows() : req.n_rows;
        if (req.all_rows && req.n_rows > 0 && req.n_rows != t.nrows())
            throw std::invalid_argument("all_rows requires n_rows equal to nrows");
        for (Index k = 1; !req.all_rows && k < count; ++k)
            if (req.rows[k] <= req.rows[k - 1]) throw std::invalid_argument("rows must be sorted and unique");
        if (count > 0 && (row_at(req, 0) < 0 || row_at(req, count - 1) >= t.nrows()))
            throw std::out_of_range(t.path() + ": row out of range");
    }

    // split into tasks: consecutive selected rows whose file span
    // does not exceed block_bytes.
    std::vector<Task> tasks;
    for (std::size_t r = 0; r < requests.size(); ++r) {
        const auto& req = requests[r];
        const Index count = req.all_rows ? req.table->nrows() : req.n_rows;
        const Index rows_per_block = std::max<Index>(1, Index(block_bytes / req.table->row_bytes()));
        Index b = 0;
        while (b < count) {
            const Index limit = row_at(req, b) + rows_per_block;  // first row beyond the block
            Index e;
            if (req.all_rows) e = std::min(count, limit);
            else e = Index(std::lower_bound(req.rows + b, req.rows + count, limit) - req.rows);
            tasks.push_back({r, b, e});
            b = e;
        }
    }

    std::vector<std::unique_ptr<File>> files;
    files.reserve(requests.size());
    for (const auto& req : requests) files.push_back(std::make_unique<File>(req.table->path()));

    std::string error;
#pragma omp parallel num_threads(resolve_threads(threads))
    {
        std::vector<char> buf;
#pragma omp for schedule(dynamic, 1)
        for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
            try {
                const Task& task = tasks[ti];
                const auto& req = requests[task.request];
                const FitsTable& t = *req.table;
                const std::size_t rb = t.row_bytes();
                const auto& tc = cols[task.request];
                const Index first = row_at(req, task.begin);
                const Index last = row_at(req, task.end - 1);
                const Index nsel = task.end - task.begin;
                const std::size_t span = std::size_t(last - first + 1) * rb;
                auto emit = [&](const char* row, Index k) {
                    const Index o = req.out_offset + k;
                    for (std::size_t c = 0; c < tc.size(); ++c)
                        outputs[c].data[o * outputs[c].stride] = decode(row + tc[c]->offset, tc[c]->type);
                };
                // dense selection (mean gap between rows <= kMaxGap): one pread
                // for the full span, since nearly every page contains a requested
                // row. sparse selection: one pread per row.
                if (span <= std::size_t(nsel) * kMaxGap) {
                    buf.resize(span);
                    files[task.request]->read_at(buf.data(), span, t.data_offset() + std::size_t(first) * rb);
                    for (Index k = task.begin; k < task.end; ++k)
                        emit(buf.data() + std::size_t(row_at(req, k) - first) * rb, k);
                } else {
                    buf.resize(rb);
                    for (Index k = task.begin; k < task.end; ++k) {
                        files[task.request]->read_at(buf.data(), rb, t.data_offset() + std::size_t(row_at(req, k)) * rb);
                        emit(buf.data(), k);
                    }
                }
            } catch (const std::exception& e) {
#pragma omp critical
                if (error.empty()) error = e.what();
            }
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
}

std::vector<std::vector<Index>> select_rows(const std::vector<const FitsTable*>& tables, const std::string& name,
                                            int threads, std::size_t block_bytes) {
    std::vector<const FitsColumn*> cols;
    std::vector<Task> tasks;  // here `request` is the table index
    for (std::size_t t = 0; t < tables.size(); ++t) {
        const FitsColumn& c = tables[t]->column(name);
        if (c.repeat != 1 || std::strchr("LBIJK", c.type) == nullptr)
            throw std::runtime_error(tables[t]->path() + ": column " + name + " is not an integer or logical scalar");
        cols.push_back(&c);
        const Index n = tables[t]->nrows();
        const Index rows_per_block = std::max<Index>(1, Index(block_bytes / tables[t]->row_bytes()));
        for (Index b = 0; b < n; b += rows_per_block) tasks.push_back({t, b, std::min(n, b + rows_per_block)});
    }

    std::vector<std::unique_ptr<File>> files;
    files.reserve(tables.size());
    for (const auto* t : tables) files.push_back(std::make_unique<File>(t->path()));

    std::vector<std::vector<Index>> found(tasks.size());
    std::string error;
#pragma omp parallel num_threads(resolve_threads(threads))
    {
        std::vector<char> buf;
#pragma omp for schedule(dynamic, 1)
        for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
            try {
                const Task& task = tasks[ti];
                const FitsTable& t = *tables[task.request];
                const FitsColumn& c = *cols[task.request];
                const std::size_t rb = t.row_bytes();
                buf.resize(std::size_t(task.end - task.begin) * rb);
                files[task.request]->read_at(buf.data(), buf.size(), t.data_offset() + std::size_t(task.begin) * rb);
                for (Index k = task.begin; k < task.end; ++k) {
                    const char* p = buf.data() + std::size_t(k - task.begin) * rb + c.offset;
                    if (c.type == 'L' ? *p == 'T' : decode(p, c.type) != 0.0) found[ti].push_back(k);
                }
            } catch (const std::exception& e) {
#pragma omp critical
                if (error.empty()) error = e.what();
            }
        }
    }
    if (!error.empty()) throw std::runtime_error(error);

    // tasks are in (table, row) order: concatenate them per table.
    std::vector<std::vector<Index>> out(tables.size());
    for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
        auto& rows = out[tasks[ti].request];
        rows.insert(rows.end(), found[ti].begin(), found[ti].end());
        std::vector<Index>().swap(found[ti]);
    }
    return out;
}

}  // namespace astra
