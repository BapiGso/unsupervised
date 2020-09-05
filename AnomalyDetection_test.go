package unsupervised

import (
	"math/rand/v2"
	"testing"
)

func TestAnomalyDetection(t *testing.T) {
	data := make([][]float64, 100)
	for i := range data {
		data[i] = []float64{
			rand.NormFloat64()*0.5 + 2,
			rand.NormFloat64()*0.5 + 2,
		}
	}

	// 添加一些异常点
	data = append(data, []float64{10.0, 10.0})
	data = append(data, []float64{-5.0, -5.0})

	threshold := 3.0
	anomalies := AnomalyDetection(data, threshold)

	if len(anomalies) != len(data) {
		t.Errorf("Expected anomalies length %d, got %d", len(data), len(anomalies))
	}

	anomalyCount := 0
	for _, isAnomaly := range anomalies {
		if isAnomaly {
			anomalyCount++
		}
	}

	if anomalyCount < 2 {
		t.Errorf("Expected at least 2 anomalies, got %d", anomalyCount)
	}
}
