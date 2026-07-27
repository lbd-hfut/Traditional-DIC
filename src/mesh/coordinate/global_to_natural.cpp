#include <dic/mesh/coordinate/global_to_natural.hpp>

#include "g2l_internal.hpp"
#include "../element/shape_func_internal.hpp"

#include <stdexcept>

namespace dic {

NaturalCoordinate global_to_natural(
    const Element& element,
    const std::vector<Node>& nodes,
    const Eigen::Vector2d& global_point)
{
    int nn = element.node_count();
    if (static_cast<int>(nodes.size()) < nn) {
        throw std::runtime_error("global_to_natural: not enough nodes for element type");
    }

    
    std::vector<double> elem_nodes(2 * nn);
    for (int i = 0; i < nn; ++i) {
        elem_nodes[2 * i]     = nodes[static_cast<std::size_t>(i)].coordinate.x();
        elem_nodes[2 * i + 1] = nodes[static_cast<std::size_t>(i)].coordinate.y();
    }

    NaturalCoordinate result;
    double gx = global_point.x();
    double gy = global_point.y();

    double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
    bool ok = false;
    int iters = 0;

    
    switch (nn) {
        case 3:
            ok = mesh::internal::solve_point_t3(
                gx, gy, elem_nodes.data(),
                result.xi, result.eta, J11, J12, J21, J22);
            iters = ok ? 1 : 0;
            break;

        case 4: {
            mesh::internal::G2LParams params;
            ok = mesh::internal::solve_point_q4(
                gx, gy, elem_nodes.data(),
                result.xi, result.eta, J11, J12, J21, J22,
                params.max_iter);
            iters = ok ? 1 : 0;  
            break;
        }

        case 8: {
            mesh::internal::G2LParams params;
            
            double xi0 = 0.0, eta0 = 0.0;
            double q4_J11 = 0.0, q4_J12 = 0.0, q4_J21 = 0.0, q4_J22 = 0.0;
            mesh::internal::solve_point_q4(
                gx, gy, elem_nodes.data(),
                xi0, eta0, q4_J11, q4_J12, q4_J21, q4_J22,
                params.max_iter);

            ok = mesh::internal::solve_point_q8(
                gx, gy, elem_nodes.data(),
                xi0, eta0,
                result.xi, result.eta, J11, J12, J21, J22,
                params.tol_global, params.max_iter);
            if (!ok) {
                ok = mesh::internal::solve_point_q8_fallback(
                    gx, gy, elem_nodes.data(),
                    xi0, eta0,
                    result.xi, result.eta, J11, J12, J21, J22,
                    params.tol_global, params.tol_local, params.max_iter);
            }
            iters = ok ? 1 : 0;
            break;
        }

        default:
            throw std::runtime_error("global_to_natural: unsupported node count");
    }

    result.converged  = ok;
    result.iterations = iters;
    return result;
}

} // namespace dic
