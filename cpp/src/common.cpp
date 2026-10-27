#include "astra/common.hpp"

#include <omp.h>

namespace astra {

int resolve_threads(int requested) {
    return requested > 0 ? requested : omp_get_max_threads();
}

bool has_parallel_triangulation() {
#ifdef CGAL_LINKED_WITH_TBB
    return true;
#else
    return false;
#endif
}

}  // namespace astra
