#pragma once

#include <tsl/robin_map.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>
#include <tuple>

namespace genz_icp {

struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    // (source, target, normals, non_planar_source, non_planar_target, planar_count, non_planar_count)
    using Vector3dVectorTuple7 = std::tuple<Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, size_t, size_t>;
    using Voxel = Eigen::Vector3i;

    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector3d> points;
        int num_points_;
        
        VoxelBlock() : num_points_(20) {} // Default constructor for map operator[]
        VoxelBlock(const Eigen::Vector3d &point, int max_points) : points(), num_points_(max_points) {
            points.reserve(static_cast<size_t>(num_points_));
            points.emplace_back(point);
        }

        inline void AddPoint(const Eigen::Vector3d &point) {
            if (points.size() < static_cast<size_t>(num_points_)) {
                points.emplace_back(point);
            }
        }
    };

    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, double map_cleanup_radius,
                          double planarity_threshold, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          map_cleanup_radius_(map_cleanup_radius),
          planarity_threshold_(planarity_threshold),
          max_points_per_voxel_(max_points_per_voxel) {}

    // Original GenZ-ICP function (kept for compatibility if needed)
    Vector3dVectorTuple7 GetCorrespondences(const Vector3dVector &points,
                                            double max_correspondance_distance) const;

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }

    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;

    // Helper to find closest neighbor and return stats (kept from GenZ)
    std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> GetClosestNeighbor(const Eigen::Vector3d &query) const;
    
    // Internal helper for GenZ
    std::pair<bool, Eigen::Vector3d> DeterminePlanarity(const Eigen::Matrix3d &covariance) const;

    // === [ADDED] New function for Adaptive Hybrid-ICP logic ===
    // Returns: {closest_point, all_neighbors_in_search_radius, distance}
    std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector3d>, double> 
    GetClosestNeighborAndNeighbors(const Eigen::Vector3d &query) const;
    // ==========================================================

    double voxel_size_;
    double max_distance_;
    double map_cleanup_radius_;
    double planarity_threshold_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};

}  // namespace genz_icp