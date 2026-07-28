#include <dic/subset/region.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dic {
namespace {

struct RegionBuildBuffer {
    std::vector<std::vector<int>> nodelist;
    int leftbound{0};
    int rightbound{0};
    int totalpoints{0};
};

struct QueueNode {
    int top{0};
    int bottom{0};
    int column{0};
};

void form_regions_impl(std::vector<SubsetRegion>& regions,
                       bool& removed,
                       const std::vector<std::uint8_t>& mask,
                       int width,
                       int height,
                       int cutoff,
                       bool preserve_length)
{
    removed = false;
    regions.clear();
    if (width <= 0 || height <= 0) {
        return;
    }
    if (static_cast<int>(mask.size()) != width * height) {
        throw std::invalid_argument("Mask size does not match width and height.");
    }

    int leftbound = width - 1;
    int rightbound = 0;
    bool firstpoint = true;

    if (preserve_length) {
        leftbound = 0;
        rightbound = width - 1;
        for (int x = 0; x < width && firstpoint; ++x) {
            for (int y = 0; y < height; ++y) {
                if (mask[y + x * height] != 0) {
                    firstpoint = false;
                    break;
                }
            }
        }
    } else {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                if (mask[y + x * height] != 0) {
                    if (firstpoint) {
                        leftbound = x;
                        firstpoint = false;
                    }
                    rightbound = x;
                    break;
                }
            }
        }
    }

    if (firstpoint) {
        return;
    }

    const int column_count = rightbound - leftbound + 1;
    std::vector<std::vector<int>> all_nodes(column_count);
    std::vector<std::vector<std::uint8_t>> active(column_count);

    for (int x = leftbound; x <= rightbound; ++x) {
        const int col = x - leftbound;
        bool in_run = false;
        int top = 0;
        for (int y = 0; y < height; ++y) {
            if (!in_run && mask[y + x * height] != 0) {
                in_run = true;
                top = y;
            }
            if (in_run && (mask[y + x * height] == 0 || y == height - 1)) {
                const int bottom = (y == height - 1 && mask[y + x * height] != 0) ? y : y - 1;
                all_nodes[col].push_back(top);
                all_nodes[col].push_back(bottom);
                active[col].push_back(1);
                in_run = false;
            }
        }
    }

    std::vector<RegionBuildBuffer> separated;
    std::vector<QueueNode> queue;

    for (int col = 0; col < column_count; ++col) {
        if (all_nodes[col].empty()) {
            continue;
        }

        for (int pair = 0; pair < static_cast<int>(active[col].size()); ++pair) {
            if (active[col][pair] == 0) {
                continue;
            }

            RegionBuildBuffer current;
            current.nodelist.resize(column_count);
            current.leftbound = preserve_length ? leftbound : col + leftbound;
            current.rightbound = preserve_length ? rightbound : 0;
            current.totalpoints = 0;

            active[col][pair] = 0;
            queue.push_back({all_nodes[col][pair * 2], all_nodes[col][pair * 2 + 1], col});

            while (!queue.empty()) {
                const QueueNode node = queue.back();
                queue.pop_back();

                auto& column_nodes = current.nodelist[node.column];
                column_nodes.push_back(node.top);
                column_nodes.push_back(node.bottom);
                std::sort(column_nodes.begin(), column_nodes.end());
                current.totalpoints += node.bottom - node.top + 1;
                if (!preserve_length) {
                    current.rightbound = std::max(current.rightbound, node.column);
                }

                const int neighbor_columns[2] = {node.column - 1, node.column + 1};
                for (const int next_col : neighbor_columns) {
                    if (next_col < 0 || next_col >= column_count) {
                        continue;
                    }
                    for (int j = 0; j < static_cast<int>(all_nodes[next_col].size()); j += 2) {
                        if (all_nodes[next_col][j] > node.bottom) {
                            break;
                        }
                        if (active[next_col][j / 2] == 0 || all_nodes[next_col][j + 1] < node.top) {
                            continue;
                        }
                        active[next_col][j / 2] = 0;
                        queue.push_back({all_nodes[next_col][j], all_nodes[next_col][j + 1], next_col});
                    }
                }
            }

            if (!preserve_length) {
                current.rightbound += leftbound;
            }

            if (current.totalpoints > cutoff) {
                separated.push_back(std::move(current));
            } else {
                removed = true;
            }
        }
    }

    regions.resize(separated.size());
    for (int i = 0; i < static_cast<int>(separated.size()); ++i) {
        int max_nodes = 0;
        for (const auto& nodes : separated[i].nodelist) {
            max_nodes = std::max(max_nodes, static_cast<int>(nodes.size()));
        }

        auto& region = regions[i];
        region.height_nodelist = separated[i].rightbound - separated[i].leftbound + 1;
        region.width_nodelist = max_nodes;
        region.nodelist.assign(region.height_nodelist * region.width_nodelist, 0);
        region.noderange.assign(region.height_nodelist, 0);
        region.leftbound = separated[i].leftbound;
        region.rightbound = separated[i].rightbound;
        region.totalpoints = separated[i].totalpoints;
        region.upperbound = height;
        region.lowerbound = 0;

        int dst_col = 0;
        for (int src_col = 0; src_col < static_cast<int>(separated[i].nodelist.size()); ++src_col) {
            const auto& nodes = separated[i].nodelist[src_col];
            if (!preserve_length && nodes.empty()) {
                continue;
            }
            region.noderange[dst_col] = static_cast<int32_t>(nodes.size());
            for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
                region.nodelist[dst_col + j * region.height_nodelist] = nodes[j];
                if (j % 2 == 0) {
                    region.upperbound = std::min(region.upperbound, nodes[j]);
                } else {
                    region.lowerbound = std::max(region.lowerbound, nodes[j]);
                }
            }
            ++dst_col;
        }
    }
}

} // namespace

bool SubsetRegion::contains(int x, int y) const
{
    const int idx_x = x - leftbound;
    if (idx_x < 0 || idx_x >= height_nodelist) {
        return false;
    }
    for (int i = 0; i < noderange[idx_x]; i += 2) {
        const int top = nodelist[idx_x + i * height_nodelist];
        const int bottom = nodelist[idx_x + (i + 1) * height_nodelist];
        if (y >= top && y <= bottom) {
            return true;
        }
    }
    return false;
}

std::vector<SubsetRegion> form_subset_regions(const std::vector<std::uint8_t>& mask,
                                              int width,
                                              int height,
                                              int cutoff,
                                              bool* removed)
{
    std::vector<SubsetRegion> regions;
    bool removed_value = false;
    form_regions_impl(regions, removed_value, mask, width, height, cutoff, false);
    if (removed != nullptr) {
        *removed = removed_value;
    }
    return regions;
}

CircularSubsetRegion extract_circular_subset(const SubsetRegion& region,
                                             int x,
                                             int y,
                                             int radius,
                                             bool truncate_subset)
{
    if (radius < 0) {
        throw std::invalid_argument("Subset radius must be non-negative.");
    }

    const int diameter = 2 * radius + 1;
    CircularSubsetRegion subset;
    subset.region.height_nodelist = diameter;
    subset.region.width_nodelist = std::max(2, region.width_nodelist);
    subset.region.leftbound = x - radius;
    subset.region.rightbound = x + radius;
    subset.region.upperbound = y + radius;
    subset.region.lowerbound = y - radius;
    subset.region.nodelist.assign(subset.region.height_nodelist * subset.region.width_nodelist, 0);
    subset.region.noderange.assign(subset.region.height_nodelist, 0);
    subset.mask.assign(diameter * diameter, 0);
    subset.radius = radius;
    subset.x = x;
    subset.y = y;

    for (int col = 0; col < diameter; ++col) {
        const int dx = col - radius;
        const double inside = static_cast<double>(radius * radius - dx * dx);
        const int top = y + static_cast<int>(std::ceil(-std::sqrt(std::max(0.0, inside))));
        const int bottom = y + static_cast<int>(std::floor(std::sqrt(std::max(0.0, inside))));

        int run_top = 0;
        bool in_run = false;
        int range = 0;
        for (int py = top; py <= bottom; ++py) {
            const bool valid = !truncate_subset || region.contains(x + dx, py);
            if (valid) {
                subset.mask[(py - (y - radius)) + col * diameter] = 1;
                subset.region.upperbound = std::min(subset.region.upperbound, py);
                subset.region.lowerbound = std::max(subset.region.lowerbound, py);
                ++subset.region.totalpoints;
            }
            if (valid && !in_run) {
                run_top = py;
                in_run = true;
            }
            if (in_run && (!valid || py == bottom)) {
                const int run_bottom = valid ? py : py - 1;
                if (range + 1 >= subset.region.width_nodelist) {
                    const int old_width = subset.region.width_nodelist;
                    subset.region.width_nodelist += 2;
                    std::vector<int32_t> resized(subset.region.height_nodelist *
                                                 subset.region.width_nodelist, 0);
                    for (int old_j = 0; old_j < old_width; ++old_j) {
                        for (int old_col = 0; old_col < subset.region.height_nodelist; ++old_col) {
                            resized[old_col + old_j * subset.region.height_nodelist] =
                                subset.region.nodelist[old_col + old_j * subset.region.height_nodelist];
                        }
                    }
                    subset.region.nodelist.swap(resized);
                }
                subset.region.nodelist[col + range * diameter] = run_top;
                subset.region.nodelist[col + (range + 1) * diameter] = run_bottom;
                range += 2;
                in_run = false;
            }
        }
        subset.region.noderange[col] = range;
    }

    if (subset.region.totalpoints == 0) {
        subset.region.upperbound = 0;
        subset.region.lowerbound = 0;
    }

    return subset;
}

} // namespace dic
