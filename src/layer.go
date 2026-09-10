package codex

type Layer struct {
	layerNumber   int
	size          int
	nodes         []Node
	previousLayer *Layer
	nextLayer     *Layer
}

func (l *Layer) GetLayerNumber() int {
	return l.layerNumber
}

func (l *Layer) AddNodes(nodes []Node) {
	l.nodes = append(l.nodes, nodes...)
	l.size += len(nodes)
}

func (l *Layer) GetSize() int {
	return l.size
}

func (l *Layer) AddNextLayer(nextLayer *Layer) {
	l.nextLayer = nextLayer
}

func (l *Layer) GetNextLayer() *Layer {
	return l.nextLayer
}

func (l *Layer) GetPreviousLayer() *Layer {
	return l.previousLayer
}
