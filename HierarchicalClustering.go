package unsupervised

import "math"

// HierarchicalClustering 实现层次聚类算法
func HierarchicalClustering(data [][]float64, k int) []int {
	n := len(data)
	if n == 0 {
		return nil
	}

	// 初始化距离矩阵
	distances := make([][]float64, n)
	for i := range distances {
		distances[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i != j {
				distances[i][j] = euclideanDistance(data[i], data[j])
			}
		}
	}

	// 初始化簇,每个点作为一个簇
	clusters := make([][]int, n)
	for i := range clusters {
		clusters[i] = []int{i}
	}

	// 合并最近的簇直到达到k个簇
	for len(clusters) > k {
		minDist := math.MaxFloat64
		var mergeI, mergeJ int

		// 找到最近的两个簇
		for i := 0; i < len(clusters); i++ {
			for j := i + 1; j < len(clusters); j++ {
				dist := clusterDistance(clusters[i], clusters[j], distances)
				if dist < minDist {
					minDist = dist
					mergeI = i
					mergeJ = j
				}
			}
		}

		// 合并簇
		clusters[mergeI] = append(clusters[mergeI], clusters[mergeJ]...)
		clusters = append(clusters[:mergeJ], clusters[mergeJ+1:]...)
	}

	// 生成结果
	result := make([]int, n)
	for i, cluster := range clusters {
		for _, idx := range cluster {
			result[idx] = i
		}
	}

	return result
}
