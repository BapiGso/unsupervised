package unsupervised

import "math/rand/v2"

// Autoencoder 实现一个简单的自编码器
type Autoencoder struct {
	inputDim  int
	hiddenDim int
	weights   [][]float64
	biases    []float64
}

func NewAutoencoder(inputDim, hiddenDim int) *Autoencoder {
	weights := make([][]float64, inputDim)
	for i := range weights {
		weights[i] = make([]float64, hiddenDim)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1
		}
	}

	biases := make([]float64, hiddenDim)
	for i := range biases {
		biases[i] = rand.Float64()*2 - 1
	}

	return &Autoencoder{
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		weights:   weights,
		biases:    biases,
	}
}

func (ae *Autoencoder) Encode(input []float64) []float64 {
	if len(input) != ae.inputDim {
		return nil
	}

	hidden := make([]float64, ae.hiddenDim)
	for i := 0; i < ae.hiddenDim; i++ {
		sum := ae.biases[i]
		for j := 0; j < ae.inputDim; j++ {
			sum += input[j] * ae.weights[j][i]
		}
		hidden[i] = sigmoid(sum)
	}
	return hidden
}
