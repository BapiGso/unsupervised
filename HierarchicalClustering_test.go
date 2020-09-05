package unsupervised

import "testing"

func TestHierarchicalClustering(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0},
		{1.2, 1.1},
		{0.8, 1.2},
		{3.7, 4.0},
		{3.9, 3.9},
		{4.0, 4.1},
	}

	k := 2
	clusters := HierarchicalClustering(data, k)

	// 验证基本属性
	if len(clusters) != len(data) {
		t.Errorf("Expected clusters length %d, got %d", len(data), len(clusters))
	}

	// 验证簇的数量
	uniqueClusters := make(map[int]bool)
	for _, c := range clusters {
		uniqueClusters[c] = true
	}
	if len(uniqueClusters) != k {
		t.Errorf("Expected %d unique clusters, got %d", k, len(uniqueClusters))
	}
}
