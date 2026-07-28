#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_PADDING_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_PADDING_HPP

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/subset/subset_config.hpp>

namespace dic {

int recommended_subset_padding(const SubsetConfig& config);
Image mirror_pad_image(const Image& image, int pad);
Mask zero_pad_mask(const Mask& mask, int pad);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_PADDING_HPP
