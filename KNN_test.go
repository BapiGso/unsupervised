package unsupervised

import "testing"

func TestKMeans(t *testing.T) {
	// 生成测试数据
	data := [][]float64{
		{1.0, 1.0},
		{1.5, 2.0},
		{3.0, 4.0},
		{5.0, 7.0},
		{3.5, 5.0},
		{4.5, 5.0},
		{3.5, 4.5},
	}

	k := 2
	maxIter := 100

	clusters, centroids := KMeans(data, k, maxIter)

	// 验证基本属性
	if len(clusters) != len(data) {
		t.Errorf("Expected clusters length %d, got %d", len(data), len(clusters))
	}

	if len(centroids) != k {
		t.Errorf("Expected centroids length %d, got %d", k, len(centroids))
	}

	// 验证簇的有效性
	for _, c := range clusters {
		if c < 0 || c >= k {
			t.Errorf("Invalid cluster assignment: %d", c)
		}
	}
}
