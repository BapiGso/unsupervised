package unsupervised

// AnomalyDetection 实现基于统计的异常检测
func AnomalyDetection(data [][]float64, threshold float64) []bool {
	if len(data) == 0 {
		return nil
	}

	n := len(data)
	dim := len(data[0])

	// 计算均值
	mean := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			mean[j] += data[i][j]
		}
	}
	for j := 0; j < dim; j++ {
		mean[j] /= float64(n)
	}

	// 计算协方差矩阵
	cov := make([][]float64, dim)
	for i := range cov {
		cov[i] = make([]float64, dim)
		for j := range cov[i] {
			for k := 0; k < n; k++ {
				cov[i][j] += (data[k][i] - mean[i]) * (data[k][j] - mean[j])
			}
			cov[i][j] /= float64(n - 1)
		}
	}

	// 计算马氏距离
	anomalies := make([]bool, n)
	for i := 0; i < n; i++ {
		dist := mahalanobisDistance(data[i], mean, cov)
		anomalies[i] = dist > threshold
	}

	return anomalies
}
