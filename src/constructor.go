package codex

import (
	"fmt"
	"math/rand"
)

type CodexConstructor struct {
	inputType      InputType
	inputSize      int
	firstLayer     *Layer
	outputLayer    *Layer
	isComplete     bool
	activationFunc func(float64) float64
}

func NewCodexConstructor() *CodexConstructor {
	return &CodexConstructor{}
}

var _ CodexConstructorModel = (*CodexConstructor)(nil)

// AddInputSize implements CodexConstructorModel
func (c *CodexConstructor) AddInputSize(n int, inputType InputType) error {
	c.inputType = inputType

	if n <= 0 {
		return fmt.Errorf("input size must be greater than 0")
	}

	c.inputSize = n

	return nil
}

type Node struct {
	weights []InputType
	biases  []InputType
	layer   *Layer
}

// AddLayer implements CodexConstructorModel
func (c *CodexConstructor) AddLayer(layerSize int) error {
	if layerSize <= 0 {
		return fmt.Errorf("layer size must be greater than 0")
	}

	currentLayerNumber := 1

	if c.firstLayer != nil {
		lastLayer, err := c.getLastLayer()
		if err != nil {
			return err
		}

		currentLayerNumber = lastLayer.GetLayerNumber() + 1
	}

	var previousLayer *Layer
	previousLayer = nil

	inputSize := c.inputSize

	// If first layer, inputChannel should be determined by inputSize and no previousLayer should be set
	if currentLayerNumber != 1 {
		previousLayer, err := c.getLayerN(c.firstLayer, currentLayerNumber-1)
		if err != nil {
			return err
		}

		inputSize = previousLayer.GetSize()
	}

	layer := &Layer{
		layerNumber:   currentLayerNumber,
		size:          0,
		nodes:         nil,
		previousLayer: previousLayer,
		nextLayer:     nil,
	}

	nodes := make([]Node, layerSize)

	for i := 0; i < layerSize; i++ {
		size := inputSize
		if previousLayer != nil {
			size = previousLayer.GetSize()
		}

		weights := make([]InputType, size)
		biases := make([]InputType, size)

		// Randomize weights and biases
		for j := 0; j < size; j++ {
			weights[j] = rand.Float64()
			biases[j] = rand.Float64()
		}

		nodes[i] = Node{
			weights: weights,
			biases:  biases,
			layer:   layer,
		}
	}

	layer.AddNodes(nodes)

	return nil

}

// AddOutputLayer implements CodexConstructorModel
func (c *CodexConstructor) AddOutputLayer(outputLayerSize int) error {
	if outputLayerSize <= 0 {
		return fmt.Errorf("output layer size must be greater than 0")
	}

	lastLayer, err := c.getLastLayer()
	if err != nil {
		return err
	}

	layer := &Layer{
		layerNumber:   lastLayer.GetLayerNumber() + 1,
		size:          0,
		nodes:         nil,
		previousLayer: lastLayer,
		nextLayer:     nil,
	}

	nodes := make([]Node, outputLayerSize)

	for i := 0; i < outputLayerSize; i++ {
		weights := make([]InputType, lastLayer.GetSize())
		biases := make([]InputType, lastLayer.GetSize())

		// Randomize weights and biases
		for j := 0; j < lastLayer.GetSize(); j++ {
			weights[j] = rand.Float64()
			biases[j] = rand.Float64()
		}

		nodes[i] = Node{
			weights: weights,
			biases:  biases,
			layer:   layer,
		}
	}

	layer.AddNodes(nodes)

	return nil
}

// AddActivationFunc implements CodexConstructorModel
func (c *CodexConstructor) AddActivationFunc(f func(float64) float64) error {
	if f == nil {
		return fmt.Errorf("activation function cannot be nil")
	}
	c.activationFunc = f
	return nil
}

// IsComplete() implements CodexConstructorModel
func (c *CodexConstructor) IsComplete() bool {
	if c.outputLayer == nil || c.firstLayer == nil {
		return false
	}

	return true
}

func (c *CodexConstructor) getLastLayer() (*Layer, error) {
	if c.firstLayer == nil {
		return nil, fmt.Errorf("no layers have been added yet")
	}

	layer := c.firstLayer
	for layer.nextLayer != nil {
		layer = layer.nextLayer
	}

	return layer, nil
}

func (c *CodexConstructor) getLayerN(layer *Layer, layerNumber int) (*Layer, error) {
	if layerNumber < 0 {
		return nil, fmt.Errorf("layer number must be greater than or equal to 0")
	}

	if layer.layerNumber == layerNumber {
		return layer, nil
	}

	if layer.nextLayer == nil {
		return nil, fmt.Errorf("layer number %d not found", layerNumber)
	}

	return c.getLayerN(layer.nextLayer, layerNumber)
}

func (c *CodexConstructor) Build() (*Codex, error) {
	codex := &Codex{
		inputSize:      c.inputSize,
		outputSize:     c.outputLayer.GetSize(),
		firstLayer:     c.firstLayer,
		outputLayer:    c.outputLayer,
		activationFunc: c.activationFunc,
	}

	return codex, nil
}
