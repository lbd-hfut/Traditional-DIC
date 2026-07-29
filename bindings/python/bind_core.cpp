/**
 * @file bind_core.cpp
 * @brief pybind11 bindings for lightweight core image/mask utilities.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

dic::Image core_image_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const float*>(buf.ptr);
    std::vector<float> data(src, src + static_cast<std::size_t>(w * h));
    return dic::Image(w, h, std::move(data));
}

dic::Image core_image_from_numpy64(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const double*>(buf.ptr);
    std::vector<float> data(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(src[i]);
    }
    return dic::Image(w, h, std::move(data));
}

py::array_t<float> image_to_numpy(const dic::Image& image)
{
    py::array_t<float> arr({image.height(), image.width()});
    auto* dst = static_cast<float*>(arr.request().ptr);
    std::copy(image.data().begin(), image.data().end(), dst);
    return arr;
}

dic::Mask core_mask_from_numpy_u8(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Mask must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const std::uint8_t*>(buf.ptr);
    std::vector<bool> data(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = src[i] != 0;
    }
    return dic::Mask(w, h, std::move(data));
}

py::array_t<std::uint8_t> mask_to_numpy(const dic::Mask& mask)
{
    py::array_t<std::uint8_t> arr({mask.height(), mask.width()});
    auto* dst = static_cast<std::uint8_t*>(arr.request().ptr);
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            dst[static_cast<std::size_t>(y * mask.width() + x)] = mask.valid(x, y) ? 1 : 0;
        }
    }
    return arr;
}

dic::Image py_image_from_object(py::object obj)
{
    if (py::isinstance<py::array_t<float>>(obj)) {
        return core_image_from_numpy(py::cast<py::array_t<float>>(obj));
    }
    if (py::isinstance<py::array_t<double>>(obj)) {
        return core_image_from_numpy64(py::cast<py::array_t<double>>(obj));
    }
    throw std::runtime_error("image must be a 2D numpy array (float32 or float64).");
}

} // namespace

void bind_core(py::module_& m)
{
    auto sub = m.def_submodule("core");

    py::enum_<dic::ImageNormalization>(sub, "ImageNormalization")
        .value("None_", dic::ImageNormalization::None)
        .value("MaxIntensity", dic::ImageNormalization::MaxIntensity)
        .value("GlobalMeanStd", dic::ImageNormalization::GlobalMeanStd)
        .value("RoiMeanStd", dic::ImageNormalization::RoiMeanStd);

    py::class_<dic::Image>(sub, "Image")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def_property_readonly("width", &dic::Image::width)
        .def_property_readonly("height", &dic::Image::height)
        .def("empty", &dic::Image::empty)
        .def("to_numpy", &image_to_numpy);

    py::class_<dic::Mask>(sub, "Mask")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def(py::init<const std::string&>())
        .def_property_readonly("width", &dic::Mask::width)
        .def_property_readonly("height", &dic::Mask::height)
        .def("empty", &dic::Mask::empty)
        .def("to_numpy", &mask_to_numpy);

    sub.def("image_from_numpy", [](py::object image) { return py_image_from_object(image); }, py::arg("image"));
    sub.def("mask_from_numpy", &core_mask_from_numpy_u8, py::arg("mask"));
    sub.def("load_image", [](const std::string& path) { return dic::Image(path); }, py::arg("path"));
    sub.def("load_mask", [](const std::string& path) { return dic::Mask(path); }, py::arg("path"));
    sub.def("normalize_image",
        [](py::object image, const std::string& method) {
            dic::ImageNormalization normalization = dic::ImageNormalization::None;
            if (method == "max" || method == "max_intensity") {
                normalization = dic::ImageNormalization::MaxIntensity;
            } else if (method == "global_mean_std") {
                normalization = dic::ImageNormalization::GlobalMeanStd;
            } else if (method != "none") {
                throw std::runtime_error("Unsupported normalization method: " + method);
            }
            return image_to_numpy(dic::normalize_image(py_image_from_object(image), normalization));
        },
        py::arg("image"),
        py::arg("method") = "none");
}
