#include <dic/subset/padding.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dic {
namespace {

int mirror_index(int index, int size)
{
    if (size <= 1) {
        return 0;
    }
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index - 1;
        } else {
            index = 2 * size - index - 1;
        }
    }
    return index;
}

} // namespace

int recommended_subset_padding(const SubsetConfig& config)
{
    const int integer_radius = std::max(1, config.seed_initialization.integer_search.subset_radius);
    const int subpixel_radius = std::max(1, config.seed_initialization.subpixel.subset_radius);
    const int max_subset_radius = std::max({config.subset_radius, integer_radius, subpixel_radius});
    const int search_radius = std::max(0, config.seed_initialization.integer_search.search_radius);
    const int bspline_border = std::max(0, config.image_precompute.border);
    return search_radius + max_subset_radius + bspline_border;
}

Image mirror_pad_image(const Image& image, int pad)
{
    if (image.empty()) {
        return image;
    }
    pad = std::max(0, pad);
    if (pad == 0) {
        return Image(image.width(), image.height(), image.data());
    }

    const int padded_width = image.width() + 2 * pad;
    const int padded_height = image.height() + 2 * pad;
    std::vector<float> data(static_cast<std::size_t>(padded_width * padded_height), 0.0F);
    for (int y = 0; y < padded_height; ++y) {
        const int src_y = mirror_index(y - pad, image.height());
        for (int x = 0; x < padded_width; ++x) {
            const int src_x = mirror_index(x - pad, image.width());
            data[static_cast<std::size_t>(y * padded_width + x)] = image.at(src_x, src_y);
        }
    }
    return Image(padded_width, padded_height, std::move(data));
}

Mask zero_pad_mask(const Mask& mask, int pad)
{
    if (mask.empty()) {
        return mask;
    }
    pad = std::max(0, pad);
    if (pad == 0) {
        std::vector<bool> data;
        data.reserve(mask.size());
        for (int y = 0; y < mask.height(); ++y) {
            for (int x = 0; x < mask.width(); ++x) {
                data.push_back(mask.valid(x, y));
            }
        }
        return Mask(mask.width(), mask.height(), std::move(data));
    }

    Mask padded(mask.width() + 2 * pad, mask.height() + 2 * pad);
    padded.fill(false);
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            if (mask.valid(x, y)) {
                padded.set(x + pad, y + pad, true);
            }
        }
    }
    return padded;
}

} // namespace dic
