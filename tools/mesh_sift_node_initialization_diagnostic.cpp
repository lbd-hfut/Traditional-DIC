#include <dic/core/image.hpp>
#include <dic/initialization/feature_matcher.hpp>
#include <dic/initialization/sift_initializer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

struct NodeRecord {
    int id{0};
    double x{0.0};
    double y{0.0};
};

struct MatchRecord {
    double x0{0.0};
    double y0{0.0};
    double x1{0.0};
    double y1{0.0};
    double u{0.0};
    double v{0.0};
    double descriptor_distance{0.0};
    bool ratio_passed{false};
    bool mutual{false};
    bool robust_inlier{false};
};

struct NodeInitRecord {
    int id{0};
    double x{0.0};
    double y{0.0};
    bool valid{false};
    double u{0.0};
    double v{0.0};
    int support_count{0};
    double nearest_distance{0.0};
};

std::vector<NodeRecord> read_nodes(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open nodes file: " + path.string());
    }

    std::vector<NodeRecord> nodes;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        NodeRecord node;
        if (iss >> node.id >> node.x >> node.y) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), mid, values.end());
    double result = *mid;
    if (values.size() % 2U == 0U) {
        const auto left = std::max_element(values.begin(), mid);
        result = 0.5 * (result + *left);
    }
    return result;
}

void robust_filter_matches(std::vector<MatchRecord>& matches, double mad_factor)
{
    std::vector<double> us;
    std::vector<double> vs;
    us.reserve(matches.size());
    vs.reserve(matches.size());
    for (const auto& match : matches) {
        if (match.mutual) {
            us.push_back(match.u);
            vs.push_back(match.v);
        }
    }
    if (us.empty()) {
        return;
    }

    const double med_u = median(us);
    const double med_v = median(vs);
    std::vector<double> abs_du;
    std::vector<double> abs_dv;
    abs_du.reserve(us.size());
    abs_dv.reserve(vs.size());
    for (double u : us) {
        abs_du.push_back(std::abs(u - med_u));
    }
    for (double v : vs) {
        abs_dv.push_back(std::abs(v - med_v));
    }
    const double sigma_u = 1.4826 * median(abs_du);
    const double sigma_v = 1.4826 * median(abs_dv);
    const double gate_u = std::max(3.0, mad_factor * sigma_u);
    const double gate_v = std::max(3.0, mad_factor * sigma_v);

    for (auto& match : matches) {
        match.robust_inlier = match.mutual &&
            std::abs(match.u - med_u) <= gate_u &&
            std::abs(match.v - med_v) <= gate_v;
    }
}

std::vector<NodeInitRecord> interpolate_nodes(
    const std::vector<NodeRecord>& nodes,
    const std::vector<MatchRecord>& matches,
    int k,
    double radius)
{
    std::vector<const MatchRecord*> inliers;
    for (const auto& match : matches) {
        if (match.robust_inlier) {
            inliers.push_back(&match);
        }
    }

    std::vector<NodeInitRecord> out;
    out.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<std::pair<double, const MatchRecord*>> nearby;
        nearby.reserve(inliers.size());
        for (const auto* match : inliers) {
            const double dx = node.x - match->x0;
            const double dy = node.y - match->y0;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d <= radius) {
                nearby.push_back({d, match});
            }
        }
        std::sort(nearby.begin(), nearby.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        if (static_cast<int>(nearby.size()) > k) {
            nearby.resize(static_cast<std::size_t>(k));
        }

        NodeInitRecord record;
        record.id = node.id;
        record.x = node.x;
        record.y = node.y;
        record.support_count = static_cast<int>(nearby.size());
        record.nearest_distance = nearby.empty() ? 0.0 : nearby.front().first;
        if (!nearby.empty()) {
            double sum_w = 0.0;
            double sum_u = 0.0;
            double sum_v = 0.0;
            for (const auto& item : nearby) {
                const double w = 1.0 / (item.first * item.first + 1.0);
                sum_w += w;
                sum_u += w * item.second->u;
                sum_v += w * item.second->v;
            }
            record.u = sum_u / sum_w;
            record.v = sum_v / sum_w;
            record.valid = true;
        }
        out.push_back(record);
    }
    return out;
}

std::vector<NodeInitRecord> interpolate_nodes_with_initializer(
    const std::vector<NodeRecord>& nodes,
    const std::vector<dic::FeatureMatch>& matches,
    const dic::SIFTInitializer& initializer,
    int interpolation_neighbors,
    double interpolation_radius)
{
    std::vector<NodeInitRecord> out;
    out.reserve(nodes.size());
    for (const auto& node : nodes) {
        const Eigen::Vector2d point(node.x, node.y);
        const auto initial = initializer.estimate_from_matches(matches, point);

        NodeInitRecord record;
        record.id = node.id;
        record.x = node.x;
        record.y = node.y;
        record.valid = initial.valid;
        record.u = initial.u;
        record.v = initial.v;
        if (initial.valid) {
            std::vector<double> distances;
            distances.reserve(matches.size());
            double nearest = std::numeric_limits<double>::infinity();
            for (const auto& match : matches) {
                if (!match.valid || !match.robust_inlier) {
                    continue;
                }
                const double distance = (point - match.reference_point).norm();
                if (distance <= interpolation_radius) {
                    distances.push_back(distance);
                    nearest = std::min(nearest, distance);
                }
            }
            record.support_count = std::min(
                static_cast<int>(distances.size()),
                std::max(1, interpolation_neighbors));
            record.nearest_distance = std::isfinite(nearest) ? nearest : 0.0;
        }
        out.push_back(record);
    }
    return out;
}

std::vector<MatchRecord> to_match_records(const std::vector<dic::FeatureMatch>& matches)
{
    std::vector<MatchRecord> out;
    out.reserve(matches.size());
    for (const auto& match : matches) {
        MatchRecord record;
        record.x0 = match.reference_point.x();
        record.y0 = match.reference_point.y();
        record.x1 = match.deformed_point.x();
        record.y1 = match.deformed_point.y();
        record.u = match.displacement.x();
        record.v = match.displacement.y();
        record.descriptor_distance = match.confidence > 0.0 ? 1.0 / match.confidence - 1.0 : 0.0;
        record.ratio_passed = match.ratio_passed;
        record.mutual = match.mutual;
        record.robust_inlier = match.robust_inlier;
        out.push_back(record);
    }
    return out;
}

void write_matches_csv(const std::filesystem::path& path, const std::vector<MatchRecord>& matches)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "x0,y0,x1,y1,u,v,descriptor_distance,ratio_passed,mutual,robust_inlier\n";
    out << std::setprecision(12);
    for (const auto& m : matches) {
        out << m.x0 << "," << m.y0 << "," << m.x1 << "," << m.y1 << ","
            << m.u << "," << m.v << "," << m.descriptor_distance << ","
            << (m.ratio_passed ? 1 : 0) << ","
            << (m.mutual ? 1 : 0) << ","
            << (m.robust_inlier ? 1 : 0) << "\n";
    }
}

void write_nodes_csv(const std::filesystem::path& path, const std::vector<NodeInitRecord>& nodes)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "node,x,y,valid,u,v,support_count,nearest_match_distance\n";
    out << std::setprecision(12);
    for (const auto& node : nodes) {
        out << node.id << "," << node.x << "," << node.y << ","
            << (node.valid ? 1 : 0) << "," << node.u << "," << node.v << ","
            << node.support_count << "," << node.nearest_distance << "\n";
    }
}

void write_summary(const std::filesystem::path& path,
                   const std::vector<MatchRecord>& matches,
                   const std::vector<NodeInitRecord>& nodes,
                   int keypoints_ref,
                   int keypoints_def,
                   double ratio,
                   int k,
                   double radius)
{
    int ratio_count = 0;
    int mutual_count = 0;
    int inlier_count = 0;
    for (const auto& m : matches) {
        ratio_count += m.ratio_passed ? 1 : 0;
        mutual_count += m.mutual ? 1 : 0;
        inlier_count += m.robust_inlier ? 1 : 0;
    }
    int node_valid = 0;
    double sum_u = 0.0;
    double sum_v = 0.0;
    double max_mag = 0.0;
    for (const auto& node : nodes) {
        if (!node.valid) {
            continue;
        }
        ++node_valid;
        sum_u += node.u;
        sum_v += node.v;
        max_mag = std::max(max_mag, std::hypot(node.u, node.v));
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "keypoints_reference=" << keypoints_ref << "\n";
    out << "keypoints_deformed=" << keypoints_def << "\n";
    out << "ratio_threshold=" << ratio << "\n";
    out << "raw_matches=" << matches.size() << "\n";
    out << "ratio_passed=" << ratio_count << "\n";
    out << "mutual_matches=" << mutual_count << "\n";
    out << "robust_inliers=" << inlier_count << "\n";
    out << "interpolation_k=" << k << "\n";
    out << "interpolation_radius=" << radius << "\n";
    out << "nodes=" << nodes.size() << "\n";
    out << "node_valid=" << node_valid << "\n";
    out << "node_invalid=" << static_cast<int>(nodes.size()) - node_valid << "\n";
    if (node_valid > 0) {
        out << "mean_node_u=" << sum_u / node_valid << "\n";
        out << "mean_node_v=" << sum_v / node_valid << "\n";
    }
    out << "max_node_magnitude=" << max_mag << "\n";
}

void write_svg(const std::filesystem::path& path,
               const std::vector<MatchRecord>& matches,
               const std::vector<NodeInitRecord>& nodes,
               int width,
               int height)
{
    double max_mag = 0.0;
    for (const auto& node : nodes) {
        if (node.valid) {
            max_mag = std::max(max_mag, std::hypot(node.u, node.v));
        }
    }
    const double scale = max_mag > 0.0 ? 25.0 / max_mag : 1.0;

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" "
        << "orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,6 L7,3 z\" fill=\"#1f77b4\"/></marker></defs>\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"none\" stroke=\"#cccccc\"/>\n";
    out << "<g stroke=\"#999999\" stroke-width=\"0.6\" opacity=\"0.35\">\n";
    for (const auto& m : matches) {
        if (m.robust_inlier) {
            out << "<line x1=\"" << m.x0 << "\" y1=\"" << m.y0
                << "\" x2=\"" << m.x1 << "\" y2=\"" << m.y1 << "\"/>\n";
        }
    }
    out << "</g>\n";
    out << "<g stroke=\"#1f77b4\" stroke-width=\"1.2\" marker-end=\"url(#arrow)\">\n";
    for (const auto& node : nodes) {
        if (!node.valid) {
            continue;
        }
        out << "<line x1=\"" << node.x << "\" y1=\"" << node.y
            << "\" x2=\"" << node.x + node.u * scale
            << "\" y2=\"" << node.y + node.v * scale << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g>\n";
    for (const auto& node : nodes) {
        out << "<circle cx=\"" << node.x << "\" cy=\"" << node.y
            << "\" r=\"" << (node.valid ? 2.0 : 3.0)
            << "\" fill=\"" << (node.valid ? "#2ca02c" : "#d62728") << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g fill=\"#ff7f0e\" opacity=\"0.75\">\n";
    for (const auto& m : matches) {
        if (m.robust_inlier) {
            out << "<circle cx=\"" << m.x0 << "\" cy=\"" << m.y0 << "\" r=\"2\"/>\n";
        }
    }
    out << "</g>\n";
    out << "</svg>\n";
}

#ifdef TRADITIONAL_DIC_HAS_OPENCV
std::vector<MatchRecord> compute_sift_matches(const cv::Mat& reference,
                                              const cv::Mat& deformed,
                                              int max_features,
                                              double ratio,
                                              double mad_factor,
                                              int& keypoints_ref,
                                              int& keypoints_def)
{
    auto sift = cv::SIFT::create(max_features);
    std::vector<cv::KeyPoint> kp_ref;
    std::vector<cv::KeyPoint> kp_def;
    cv::Mat desc_ref;
    cv::Mat desc_def;
    sift->detectAndCompute(reference, cv::noArray(), kp_ref, desc_ref);
    sift->detectAndCompute(deformed, cv::noArray(), kp_def, desc_def);
    keypoints_ref = static_cast<int>(kp_ref.size());
    keypoints_def = static_cast<int>(kp_def.size());
    if (desc_ref.empty() || desc_def.empty()) {
        return {};
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> ref_to_def;
    std::vector<std::vector<cv::DMatch>> def_to_ref;
    matcher.knnMatch(desc_ref, desc_def, ref_to_def, 2);
    matcher.knnMatch(desc_def, desc_ref, def_to_ref, 2);

    std::vector<int> reverse_best(desc_def.rows, -1);
    for (const auto& pair : def_to_ref) {
        if (pair.size() < 2U) {
            continue;
        }
        if (pair[0].distance < ratio * pair[1].distance) {
            reverse_best[pair[0].queryIdx] = pair[0].trainIdx;
        }
    }

    std::vector<MatchRecord> matches;
    for (const auto& pair : ref_to_def) {
        if (pair.size() < 2U) {
            continue;
        }
        const auto& best = pair[0];
        const bool ratio_passed = best.distance < ratio * pair[1].distance;
        if (!ratio_passed) {
            continue;
        }
        const bool mutual = best.trainIdx >= 0 &&
            best.trainIdx < static_cast<int>(reverse_best.size()) &&
            reverse_best[best.trainIdx] == best.queryIdx;
        const auto& p0 = kp_ref[best.queryIdx].pt;
        const auto& p1 = kp_def[best.trainIdx].pt;
        MatchRecord record;
        record.x0 = p0.x;
        record.y0 = p0.y;
        record.x1 = p1.x;
        record.y1 = p1.y;
        record.u = record.x1 - record.x0;
        record.v = record.y1 - record.y0;
        record.descriptor_distance = best.distance;
        record.ratio_passed = true;
        record.mutual = mutual;
        matches.push_back(record);
    }
    robust_filter_matches(matches, mad_factor);
    return matches;
}
#endif

} // namespace

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::cerr << "Usage: mesh_sift_node_initialization_diagnostic <ref.bmp> <def.bmp>"
                  << " <nodes.txt> <out_dir> <label> [max_features=4000]"
                  << " [ratio=0.75] [k=8] [radius=150] [mad_factor=5]\n";
        return EXIT_FAILURE;
    }

#ifndef TRADITIONAL_DIC_HAS_OPENCV
    (void)argv;
    std::cerr << "mesh_sift_node_initialization_diagnostic requires OpenCV with SIFT support.\n";
    return EXIT_FAILURE;
#else
    try {
        const std::filesystem::path ref_path = argv[1];
        const std::filesystem::path def_path = argv[2];
        const std::filesystem::path nodes_path = argv[3];
        const std::filesystem::path out_dir = argv[4];
        const std::string label = argv[5];
        const int max_features = argc >= 7 ? std::atoi(argv[6]) : 4000;
        const double ratio = argc >= 8 ? std::atof(argv[7]) : 0.75;
        const int k = argc >= 9 ? std::atoi(argv[8]) : 8;
        const double radius = argc >= 10 ? std::atof(argv[9]) : 150.0;
        const double mad_factor = argc >= 11 ? std::atof(argv[10]) : 5.0;

        std::filesystem::create_directories(out_dir);

        const dic::Image reference(ref_path.string());
        const dic::Image deformed(def_path.string());
        if (reference.empty() || deformed.empty()) {
            throw std::runtime_error("Failed to load input images.");
        }
        if (reference.width() != deformed.width() || reference.height() != deformed.height()) {
            throw std::runtime_error("Reference and deformed images must have the same size.");
        }

        const auto nodes = read_nodes(nodes_path);
        dic::FeatureMatcherConfig matcher_config;
        matcher_config.max_features = max_features;
        matcher_config.ratio_threshold = ratio;
        matcher_config.robust_mad_factor = mad_factor;
        const dic::FeatureMatcher matcher(matcher_config);
        const auto feature_matches = matcher.match(reference, deformed);
        const auto matches = to_match_records(feature_matches);

        dic::SIFTInitializerConfig initializer_config;
        initializer_config.matcher = matcher_config;
        initializer_config.interpolation_neighbors = std::max(1, k);
        initializer_config.interpolation_radius = std::max(1.0, radius);
        const dic::SIFTInitializer initializer(initializer_config);
        const auto node_init = interpolate_nodes_with_initializer(
            nodes,
            feature_matches,
            initializer,
            initializer_config.interpolation_neighbors,
            initializer_config.interpolation_radius);

        write_matches_csv(out_dir / (label + "_sift_matches.csv"), matches);
        write_nodes_csv(out_dir / (label + "_sift_node_initialization.csv"), node_init);
        write_summary(out_dir / (label + "_sift_summary.txt"),
                      matches, node_init, 0, 0, ratio, k, radius);
        write_svg(out_dir / (label + "_sift_node_initialization.svg"),
                  matches, node_init, reference.width(), reference.height());

        std::cout << "Wrote SIFT node initialization diagnostics to " << out_dir.string() << "\n";
        std::cout << "Keypoints reference: 0\n";
        std::cout << "Keypoints deformed: 0\n";
        std::cout << "Matches: " << matches.size() << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_sift_node_initialization_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
#endif
}
