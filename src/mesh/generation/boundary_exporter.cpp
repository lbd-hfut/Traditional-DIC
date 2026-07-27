#include <dic/mesh/generation/boundary_exporter.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace dic::mesh {
namespace {

using PointKey = std::pair<int, int>;
using EdgeKey = std::pair<PointKey, PointKey>;

bool is_valid_or_inside(const Mask& mask, int x, int y)
{
    return mask.valid(x, y);
}

void add_edge(
    std::map<PointKey, std::vector<PointKey>>& adjacency,
    const PointKey& start,
    const PointKey& end)
{
    adjacency[start].push_back(end);
}

bool is_collinear(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    const double abx = b.x() - a.x();
    const double aby = b.y() - a.y();
    const double bcx = c.x() - b.x();
    const double bcy = c.y() - b.y();
    return std::abs(abx * bcy - aby * bcx) < 1e-12;
}

void remove_collinear_vertices(std::vector<Eigen::Vector2d>& points)
{
    if (points.size() < 4U) {
        return;
    }

    bool changed = true;
    while (changed && points.size() >= 4U) {
        changed = false;
        std::vector<Eigen::Vector2d> simplified;
        simplified.reserve(points.size());
        const std::size_t n = points.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& prev = points[(i + n - 1U) % n];
            const auto& curr = points[i];
            const auto& next = points[(i + 1U) % n];
            if (is_collinear(prev, curr, next)) {
                changed = true;
                continue;
            }
            simplified.push_back(curr);
        }
        points = std::move(simplified);
    }
}

} // namespace

std::vector<BoundaryLoop> extract_boundary_loops(const Mask& mask)
{
    if (mask.empty()) {
        throw std::invalid_argument("Cannot extract boundary from an empty mask.");
    }

    std::map<PointKey, std::vector<PointKey>> adjacency;
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            if (!mask.valid(x, y)) {
                continue;
            }

            if (!is_valid_or_inside(mask, x, y - 1)) {
                add_edge(adjacency, {x, y}, {x + 1, y});
            }
            if (!is_valid_or_inside(mask, x + 1, y)) {
                add_edge(adjacency, {x + 1, y}, {x + 1, y + 1});
            }
            if (!is_valid_or_inside(mask, x, y + 1)) {
                add_edge(adjacency, {x + 1, y + 1}, {x, y + 1});
            }
            if (!is_valid_or_inside(mask, x - 1, y)) {
                add_edge(adjacency, {x, y + 1}, {x, y});
            }
        }
    }

    std::set<EdgeKey> used;
    std::vector<BoundaryLoop> loops;
    for (const auto& [start, ends] : adjacency) {
        for (const auto& first_end : ends) {
            EdgeKey first_edge{start, first_end};
            if (used.find(first_edge) != used.end()) {
                continue;
            }

            std::vector<Eigen::Vector2d> points;
            PointKey current_start = start;
            PointKey current_end = first_end;
            points.emplace_back(static_cast<double>(current_start.first),
                                static_cast<double>(current_start.second));

            while (true) {
                used.insert({current_start, current_end});
                points.emplace_back(static_cast<double>(current_end.first),
                                    static_cast<double>(current_end.second));

                if (current_end == start) {
                    points.pop_back();
                    break;
                }

                auto it = adjacency.find(current_end);
                if (it == adjacency.end()) {
                    break;
                }

                bool advanced = false;
                for (const auto& candidate : it->second) {
                    EdgeKey edge{current_end, candidate};
                    if (used.find(edge) == used.end()) {
                        current_start = current_end;
                        current_end = candidate;
                        advanced = true;
                        break;
                    }
                }
                if (!advanced) {
                    break;
                }
            }

            remove_collinear_vertices(points);
            if (points.size() >= 3U) {
                loops.push_back(BoundaryLoop{std::move(points)});
            }
        }
    }

    std::sort(loops.begin(), loops.end(), [](const BoundaryLoop& a, const BoundaryLoop& b) {
        return a.points.size() > b.points.size();
    });
    return loops;
}

} // namespace dic::mesh
