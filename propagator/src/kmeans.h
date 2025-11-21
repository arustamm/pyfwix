#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

// Function to calculate the Euclidean distance between two points
double calculateDistance(std::complex<float> point1, std::complex<float> point2) {
    return std::abs(point1 - point2);
}

// Function to calculate the relative movement of cluster centers
float calculateRelativeMovement(const std::vector<std::complex<float>>& centers, const std::vector<std::complex<float>>& newCenters) {
    float movement = 0.f;
    for (int i = 0; i < centers.size(); ++i) {
        movement += std::abs((centers[i] - newCenters[i]) / centers[i]);
    }
    return movement;
}

// Function to perform k-means clustering
std::pair<std::vector<std::complex<float>>, std::vector<std::vector<int>>> kMeansClustering(const std::vector<std::complex<float>>& data, int numClusters) {

double compactness = kmeans(points, clusterCount, labels,
TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-12),
    3, KMEANS_PP_CENTERS, centers);

}

