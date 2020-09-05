package unsupervised

import "testing"

func TestAutoencoder(t *testing.T) {
	inputDim := 4
	hiddenDim := 2

	// 创建自编码器
	ae := NewAutoencoder(inputDim, hiddenDim)

	// 测试输入
	input := []float64{0.5, 0.3, 0.8, 0.1}
	encoded := ae.Encode(input)

	// 验证编码维度
	if len(encoded) != hiddenDim {
		t.Errorf("Expected encoded dimension %d, got %d", hiddenDim, len(encoded))
	}

	// 验证编码值在有效范围内 (sigmoid输出应在0-1之间)
	for _, v := range encoded {
		if v < 0 || v > 1 {
			t.Errorf("Encoded value %f outside valid range [0,1]", v)
		}
	}
}
