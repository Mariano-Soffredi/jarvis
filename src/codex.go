package codex

type Codex struct {
	inputSize      int
	outputSize     int
	firstLayer     *Layer
	outputLayer    *Layer
	activationFunc func(float64) float64
}

var _ CodexModel = (*Codex)(nil)

// Alter() implements CodexModel
func (c *Codex) Alter(growthRate float64) error {
	// TODO: To implement
	return nil
}

// Run() implements CodexModel
func (c *Codex) Run(input []float64) []float64 {
	// TODO: To implement
	return nil
}

// InputSize() implements CodexModel
func (c *Codex) InputSize() int {
	return c.inputSize
}

// OutputSize() implements CodexModel
func (c *Codex) OutputSize() int {
	return c.outputSize
}
