package codex

type InputType any

type CodexConstructorModel interface {
	// AddInputSize() adds the input size to the model
	AddInputSize(inputSize int, inputType InputType) error
	// AddLayer() adds a layer to the model with size layerSize, requires an input size to be defined first
	AddLayer(layerSize int) error
	// AddOutputLayer() adds an output layer to the model with size outputLayerSize and requires at least one layer
	AddOutputLayer(outputLayerSize int) error
	// AddActivationFunc() adds an activation function to the model
	AddActivationFunc(f func(float64) float64) error
	// IsComplete() returns true if the model is complete and ready to be built
	IsComplete() bool
	// Build() builds the model and returns a CodexModel
	Build() (*Codex, error)
}

type CodexModel interface {
	// Alter changes the nodes values randomly by the growthRate provided
	Alter(growthRate float64) error
	// Run executes the AI model, receives the input previously defined in size
	Run(input []float64) []float64
	// InputSize returns the required size of the input layer
	InputSize() int
	// OutputSize returns the size of the output layer
	OutputSize() int
}
