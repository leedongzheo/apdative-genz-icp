#include "Registration.hpp"

#include <Eigen/Eigenvalues>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>
#include <iostream>
#include <iomanip> 

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

// --- Hybrid Correspondence Struct ---
struct HybridCorrespondence {
    std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
    std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
    size_t planar_count = 0, non_planar_count = 0;
};

// --- Adaptive Threshold Functions ---

// [SỬA 1] Thêm tham số base vào hàm, không fix cứng nữa
double ComputeAdaptivePlaharityThreshold(const std::vector<Eigen::Vector3d>& neighbors, double base, double min_thr, double max_thr){
    // constexpr double min_thr = 0.001;
    // constexpr double max_thr = 0.2;
    constexpr double ref_neighbors = 20.0;

    double thr = base * ref_neighbors / std::max(ref_neighbors, static_cast<double>(neighbors.size()));
    return std::clamp(thr, min_thr, max_thr); // Dùng biến truyền vào;
}

// [SỬA 2] Nhận adaptive_base và truyền tiếp
std::tuple<bool, Eigen::Vector3d> EstimateNormalAndPlanarity(
    const std::vector<Eigen::Vector3d>& neighbors, 
    double threshold_param, // Biến này có thể là base hoặc fixed threshold tùy cờ
    bool use_adaptive,
    double min_thr, // Mới
    double max_thr  // Mới
    )
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& pt : neighbors) mean += pt;
    mean /= static_cast<double>(neighbors.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pt : neighbors) {
        Eigen::Vector3d d = pt - mean;
        cov.noalias() += d * d.transpose();
    }
    cov /= static_cast<double>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    const auto& evals = eig.eigenvalues();
    const auto& evecs = eig.eigenvectors();

    const double lambda0 = evals(0);
    const double sumlam  = evals(0) + evals(1) + evals(2) + 1e-12;
    const double planarity = lambda0 / sumlam;

    // Truyền adaptive_base vào hàm tính threshold
    // double adaptive_thr = ComputeAdaptivePlaharityThreshold(neighbors, adaptive_base);
    double final_threshold;
    if (use_adaptive) {
        // Truyền min/max vào hàm tính toán
        final_threshold = ComputeAdaptivePlaharityThreshold(neighbors, threshold_param, min_thr, max_thr);
    } else {
        final_threshold = threshold_param;
    }
    const bool is_planar = planarity < final_threshold;
    Eigen::Vector3d normal = evecs.col(0);
    return {is_planar, normal};
}

// --- Parallel Hybrid Correspondence Search ---
// [SỬA 3] Thêm tham số adaptive_base vào đây
HybridCorrespondence ComputeHybridCorrespondencesParallel(
    const std::vector<Eigen::Vector3d>& source_points,
    const genz_icp::VoxelHashMap& voxel_map,
    double max_correspondence_distance,
    double threshold_param, // base hoặc fixed
    bool use_adaptive,
    double min_thr, // Mới
    double max_thr  // Mới
    )
{
    struct LocalBuf {
        std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
        std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
        size_t planar_count = 0, non_planar_count = 0;

        void reserve_hint(size_t n) {
            const size_t hint = std::max<size_t>(32, n / 2);
            src_planar.reserve(hint);  tgt_planar.reserve(hint); normals.reserve(hint);
            src_non_planar.reserve(hint); tgt_non_planar.reserve(hint);
        }
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;
    for (auto it = tls.begin(); it != tls.end(); ++it) {
        it->reserve_hint(source_points.size());
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, source_points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& buf = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pt = source_points[i];
                
                // Call the NEW function in VoxelHashMap
                auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
                
                if (dist > max_correspondence_distance) continue;

                if (neighbors.size() >= 5) { // Min neighbors for PCA
                    // [SỬA 4] Truyền adaptive_base vào hàm estimate
                    auto [is_planar, normal] = EstimateNormalAndPlanarity(neighbors, threshold_param, use_adaptive, min_thr, max_thr);
                    if (is_planar) {
                        buf.src_planar.push_back(pt);
                        buf.tgt_planar.push_back(closest);
                        buf.normals.push_back(normal);
                        buf.planar_count++;
                    } else {
                        buf.src_non_planar.push_back(pt);
                        buf.tgt_non_planar.push_back(closest);
                        buf.non_planar_count++;
                    }
                } else {
                    // Fallback to point-to-point if not enough neighbors
                    buf.src_non_planar.push_back(pt);
                    buf.tgt_non_planar.push_back(closest);
                    buf.non_planar_count++;
                }
            }
        }
    );

    // Merge results
    HybridCorrespondence out;
    size_t total_planar = 0, total_nonplanar = 0;
    for (auto& buf : tls) {
        total_planar    += buf.planar_count;
        total_nonplanar += buf.non_planar_count;
    }
    out.src_planar.reserve(total_planar);
    out.tgt_planar.reserve(total_planar);
    out.normals.reserve(total_planar);
    out.src_non_planar.reserve(total_nonplanar);
    out.tgt_non_planar.reserve(total_nonplanar);

    for (auto& buf : tls) {
        out.planar_count     += buf.planar_count;
        out.non_planar_count += buf.non_planar_count;
        out.src_planar.insert(out.src_planar.end(), buf.src_planar.begin(), buf.src_planar.end());
        out.tgt_planar.insert(out.tgt_planar.end(), buf.tgt_planar.begin(), buf.tgt_planar.end());
        out.normals.insert(out.normals.end(), buf.normals.begin(), buf.normals.end());
        out.src_non_planar.insert(out.src_non_planar.end(), buf.src_non_planar.begin(), buf.src_non_planar.end());
        out.tgt_non_planar.insert(out.tgt_non_planar.end(), buf.tgt_non_planar.begin(), buf.tgt_non_planar.end());
    }
    return out;
}

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

// --- GenZ-ICP BuildLinearSystem (Unchanged) ---
std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    const std::vector<Eigen::Vector3d> &src_planar,
    const std::vector<Eigen::Vector3d> &tgt_planar,
    const std::vector<Eigen::Vector3d> &normals,
    const std::vector<Eigen::Vector3d> &src_non_planar,
    const std::vector<Eigen::Vector3d> &tgt_non_planar,
    double kernel,
    double alpha) {

    struct ResultTuple {
        Eigen::Matrix6d JTJ;
        Eigen::Vector6d JTr;

        ResultTuple() : JTJ(Eigen::Matrix6d::Zero()), JTr(Eigen::Vector6d::Zero()) {}

        ResultTuple operator+(const ResultTuple &other) const {
            ResultTuple result;
            result.JTJ = JTJ + other.JTJ;
            result.JTr = JTr + other.JTr;
            return result;
        }
    };

    auto compute_jacobian_and_residual_planar = [&](auto i) {
        double r_planar = (src_planar[i] - tgt_planar[i]).dot(normals[i]); 
        Eigen::Matrix<double, 1, 6> J_planar; 
        J_planar.block<1, 3>(0, 0) = normals[i].transpose(); 
        J_planar.block<1, 3>(0, 3) = (src_planar[i].cross(normals[i])).transpose();
        return std::make_tuple(J_planar, r_planar);
    };

    auto compute_jacobian_and_residual_non_planar = [&](auto i) {
        const Eigen::Vector3d r_non_planar = src_non_planar[i] - tgt_non_planar[i];
        Eigen::Matrix3_6d J_non_planar;
        J_non_planar.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_non_planar.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(src_non_planar[i]);
        return std::make_tuple(J_non_planar, r_non_planar);
    };

    double kernel_squared = kernel * kernel;
    auto compute = [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
        auto Weight = [&](double residual_squared) {
            return kernel_squared / square(kernel + residual_squared);
        };
        auto &[JTJ_private, JTr_private] = J;
        for (size_t i = r.begin(); i < r.end(); ++i) {
            if (i < src_planar.size()) { 
                const auto &[J_planar, r_planar] = compute_jacobian_and_residual_planar(i);
                double w_planar = Weight(r_planar * r_planar);
                JTJ_private.noalias() += alpha * J_planar.transpose() * w_planar * J_planar;
                JTr_private.noalias() += alpha * J_planar.transpose() * w_planar * r_planar;
            } else { 
                size_t index = i - src_planar.size();
                if (index < src_non_planar.size()) {
                    const auto &[J_non_planar, r_non_planar] = compute_jacobian_and_residual_non_planar(index);
                    const double w_non_planar = Weight(r_non_planar.squaredNorm());
                    JTJ_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * J_non_planar;
                    JTr_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * r_non_planar;
                }
            }
        }
        return J;
    };


    size_t total_size = src_planar.size() + src_non_planar.size();
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, total_size),
        ResultTuple(),
        compute,
        [](const ResultTuple &a, const ResultTuple &b) {
            return a + b;
        });

    return std::make_tuple(JTJ, JTr);
}

}  // namespace

namespace genz_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion)
    : max_num_iterations_(max_num_iteration), 
      convergence_criterion_(convergence_criterion) {}

// [SỬA 5] Cập nhật hàm RegisterFrame để nhận tham số adaptive_base (khớp với .hpp)
std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> Registration::RegisterFrame(
                                                                                                    const std::vector<Eigen::Vector3d> &frame,
                                                                                                    const VoxelHashMap &voxel_map,
                                                                                                    const Sophus::SE3d &initial_guess,
                                                                                                    double max_correspondence_distance,
                                                                                                    double kernel,
                                                                                                    double adaptive_base,
                                                                                                    bool use_adaptive,
                                                                                                    double min_thr, // Nhận vào
                                                                                                    double max_thr  // Nhận vào
                                                                                                ) { // <--- Nhận tham số ở đây
    
    std::vector<Eigen::Vector3d> final_planar_points;
    std::vector<Eigen::Vector3d> final_non_planar_points;
    final_planar_points.clear();
    final_non_planar_points.clear();

    if (voxel_map.Empty()) return std::make_tuple(initial_guess, final_planar_points, final_non_planar_points);

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        
        // [SỬA 6] Truyền adaptive_base vào hàm tìm kiếm tương ứng
        auto corr = ComputeHybridCorrespondencesParallel(
            source, 
            voxel_map, 
            max_correspondence_distance, 
            adaptive_base, // Truyền tham số này (nó chứa giá trị threshold cần dùng)
            use_adaptive,   // Truyền cờ
            min_thr, // Truyền đi
            max_thr  // Truyền đi
        );

        // double total_points = static_cast<double>(corr.planar_count + corr.non_planar_count);
        // double alpha = (total_points > 0.0) ? static_cast<double>(corr.planar_count) / total_points : 0.5;
        double alpha = 1;
        // Feed data to the solver
        const auto &[JTJ, JTr] = BuildLinearSystem(
            corr.src_planar, corr.tgt_planar, corr.normals, 
            corr.src_non_planar, corr.tgt_non_planar, 
            kernel, alpha
        );

        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        TransformPoints(estimation, source);
        T_icp = estimation * T_icp;

        if (dx.norm() < convergence_criterion_ || j == max_num_iterations_ - 1) {
            final_planar_points = corr.src_planar;
            final_non_planar_points = corr.src_non_planar;
            break;
        }
    }

    return std::make_tuple(T_icp * initial_guess, final_planar_points, final_non_planar_points);
}

}  // namespace genz_icp
