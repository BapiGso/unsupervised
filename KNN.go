package unsupervised

import (
	"math"
	"math/rand"
)

// KMeans 实现K均值聚类算法
func KMeans(data [][]float64, k int, maxIter int) ([]int, [][]float64) {
	if len(data) == 0 || k <= 0 {
		return nil, nil
	}

	n := len(data)
	dim := len(data[0])
	clusters := make([]int, n)
	centroids := make([][]float64, k)

	// 随机初始化质心
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		idx := rand.Intn(n)
		copy(centroids[i], data[idx])
	}

	for iter := 0; iter < maxIter; iter++ {
		// 分配样本到最近的质心
		changed := false
		for i := 0; i < n; i++ {
			minDist := math.MaxFloat64
			oldCluster := clusters[i]

			for j := 0; j < k; j++ {
				dist := euclideanDistance(data[i], centroids[j])
				if dist < minDist {
					minDist = dist
					clusters[i] = j
				}
			}

			if oldCluster != clusters[i] {
				changed = true
			}
		}

		if !changed {
			break
		}

		// 更新质心
		for i := 0; i < k; i++ {
			var count int
			newCentroid := make([]float64, dim)

			for j := 0; j < n; j++ {
				if clusters[j] == i {
					for d := 0; d < dim; d++ {
						newCentroid[d] += data[j][d]
					}
					count++
				}
			}

			if count > 0 {
				for d := 0; d < dim; d++ {
					centroids[i][d] = newCentroid[d] / float64(count)
				}
			}
		}
	}

	return clusters, centroids
}

// 辅助函数
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func clusterDistance(c1, c2 []int, distances [][]float64) float64 {
	minDist := math.MaxFloat64
	for _, i := range c1 {
		for _, j := range c2 {
			if distances[i][j] < minDist {
				minDist = distances[i][j]
			}
		}
	}
	return minDist
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func mahalanobisDistance(x, mean []float64, cov [][]float64) float64 {
	dim := len(x)
	diff := make([]float64, dim)
	for i := range diff {
		diff[i] = x[i] - mean[i]
	}

	// 计算马氏距离
	var dist float64
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			dist += diff[i] * diff[j] / cov[i][j]
		}
	}
	return math.Sqrt(dist)
}
