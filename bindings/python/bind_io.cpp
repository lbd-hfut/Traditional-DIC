/**
 * @file bind_io.cpp
 * @brief pybind11 bindings for simple DIC result CSV I/O.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<double> array_to_vector(py::dict result, const char* key, std::size_t expected = 0)
{
    if (!result.contains(key)) {
        throw std::runtime_error(std::string("result is missing key: ") + key);
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> arr = py::cast<py::array_t<double>>(result[key]);
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error(std::string("result key must be a 1D array: ") + key);
    }
    if (expected != 0U && static_cast<std::size_t>(buf.shape[0]) != expected) {
        throw std::runtime_error(std::string("result array length mismatch at key: ") + key);
    }
    const auto* src = static_cast<const double*>(buf.ptr);
    return {src, src + static_cast<std::size_t>(buf.shape[0])};
}

void save_displacement_csv(py::dict result, const std::string& path)
{
    const auto x = array_to_vector(result, "x");
    const auto y = array_to_vector(result, "y", x.size());
    const auto u = array_to_vector(result, "u", x.size());
    const auto v = array_to_vector(result, "v", x.size());

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write CSV: " + path);
    }
    out << "node,x,y,u,v,mag\n";
    out.precision(12);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out << (i + 1U) << "," << x[i] << "," << y[i] << ","
            << u[i] << "," << v[i] << "," << std::hypot(u[i], v[i]) << "\n";
    }
}

py::dict load_displacement_csv(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open CSV: " + path);
    }

    std::string line;
    std::getline(in, line);
    std::vector<double> x, y, u, v, mag;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double node = 0.0;
        double xi = 0.0;
        double yi = 0.0;
        double ui = 0.0;
        double vi = 0.0;
        double mi = 0.0;
        if (iss >> node >> xi >> yi >> ui >> vi >> mi) {
            x.push_back(xi);
            y.push_back(yi);
            u.push_back(ui);
            v.push_back(vi);
            mag.push_back(mi);
        }
    }

    auto make = [](const std::vector<double>& values) {
        py::array_t<double> arr(values.size());
        auto* dst = static_cast<double*>(arr.request().ptr);
        std::copy(values.begin(), values.end(), dst);
        return arr;
    };

    py::dict result;
    result["x"] = make(x);
    result["y"] = make(y);
    result["u"] = make(u);
    result["v"] = make(v);
    result["mag"] = make(mag);
    return result;
}

} // namespace

void bind_io(py::module_& m)
{
    auto sub = m.def_submodule("io");
    sub.def("save_displacement_csv", &save_displacement_csv, py::arg("result"), py::arg("path"));
    sub.def("load_displacement_csv", &load_displacement_csv, py::arg("path"));
}
