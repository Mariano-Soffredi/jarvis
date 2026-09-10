// Package codex builds small feed-forward networks that score a variable-length
// set of items - stocks, each described by a vector of features - and return one
// raw valuation per item.
//
// There is no gradient descent here. Nothing computes a derivative and there is
// no Train. A model is improved by mutating it: Mutated returns a perturbed
// copy, the caller scores a population of them against its own fitness function,
// keeps the winners and repeats.
//
// Every weight is shared across items, so the parameter count is independent of
// how many items a model scores. One model handles 20 stocks or 2000 without
// rebuilding, and the output length always matches the input length:
//
//	features -> encoder -> e_i          per item, shared weights
//	           market = mean(e_1..e_n)
//	           [e_i || market] -> mix   per item, shared weights
//	           -> head -> 1 scalar
//
// Items influence each other through that market average and nowhere else, which
// makes Score permutation-equivariant to floating-point precision: reorder the
// input and the outputs follow.
package codex

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// Activation is applied to every hidden layer, one element at a time.
//
// Seeing one element at a time is what rules out softmax and anything else
// needing a whole layer, which is deliberate: the head's final layer is left
// linear so valuations stay unbounded, and normalising them is the caller's job.
type Activation func(float64) float64

// ReLU is max(x, 0). It is unbounded above, and a unit it drives to zero has no
// gradient to revive it, so a mutation-trained net can strand one dead for good -
// on a narrow model that happens often enough to matter. It is about 1.4x faster
// than the default math.Tanh - see BenchmarkScoreTanh and BenchmarkScoreReLU -
// so prefer it only on models wide enough to absorb the risk.
func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// maxDim keeps in*out representable: two dims at this bound multiply to 2^62,
// well inside int64. Without it New(Config{FeaturesPerStock: 1<<62, ...}) wraps
// the product to zero and returns a structurally corrupt model reporting no
// error. This is not a memory guard - a model that large dies on allocation,
// which is loud enough.
const maxDim = 1 << 31

// Config describes a model's shape. Nothing here scales with the number of items
// scored.
type Config struct {
	// FeaturesPerStock is the length of every row Score is given.
	FeaturesPerStock int
	// Encoder holds the hidden widths mapping one item's features to its
	// embedding. The final entry is the embedding width.
	Encoder []int
	// Mix is the width of the layer combining an item's embedding with the
	// market average.
	Mix int
	// Head holds the hidden widths between the mix layer and the single output
	// valuation. May be empty.
	Head []int
	// Hidden defaults to math.Tanh, which cannot produce dead units and keeps
	// activations bounded as mutation drives weights outward.
	Hidden Activation
}

// dense is one fully-connected layer. w is row-major: out rows of in columns.
type dense struct {
	w   []float64
	b   []float64
	in  int
	out int
}

// forward returns the prefix of scratch it wrote. A nil act leaves it linear.
func (d *dense) forward(scratch, src []float64, act Activation) []float64 {
	dst := scratch[:d.out]
	src = src[:d.in]
	for o := range dst {
		row := d.w[o*d.in : (o+1)*d.in]
		sum := d.b[o]
		for i, x := range src {
			sum += row[i] * x
		}
		dst[o] = sum
	}
	if act != nil {
		for i, v := range dst {
			dst[i] = act(v)
		}
	}
	return dst
}

// Net is a model, built by New. Score panics on a zero Net and SetParams
// rejects one; the other methods report an empty model.
type Net struct {
	features int
	enc      []dense
	mix      dense
	head     []dense
	act      Activation
}

// New validates cfg and returns a model with weights drawn from rng.
//
// Draw a whole population from one rng rather than one generator per model:
// separate generators seeded alike hand every member identical weights, leaving
// selection nothing to act on.
func New(cfg Config, rng *rand.Rand) (*Net, error) {
	if cfg.FeaturesPerStock <= 0 || cfg.FeaturesPerStock > maxDim {
		return nil, fmt.Errorf("codex: FeaturesPerStock must be in 1..%d, got %d", maxDim, cfg.FeaturesPerStock)
	}
	if len(cfg.Encoder) == 0 {
		return nil, fmt.Errorf("codex: Encoder must have at least one layer")
	}
	if err := checkDims("Encoder", cfg.Encoder); err != nil {
		return nil, err
	}
	if cfg.Mix <= 0 || cfg.Mix > maxDim {
		return nil, fmt.Errorf("codex: Mix must be in 1..%d, got %d", maxDim, cfg.Mix)
	}
	if err := checkDims("Head", cfg.Head); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, fmt.Errorf("codex: rng must not be nil")
	}

	act := cfg.Hidden
	if act == nil {
		act = math.Tanh
	}

	c := &Net{features: cfg.FeaturesPerStock, act: act}
	c.enc = stack(append([]int{cfg.FeaturesPerStock}, cfg.Encoder...), rng)

	embed := cfg.Encoder[len(cfg.Encoder)-1]
	// The mix layer reads an embedding concatenated with the market average.
	c.mix = newDense(2*embed, cfg.Mix, rng)

	c.head = stack(append(append([]int{cfg.Mix}, cfg.Head...), 1), rng)
	return c, nil
}

func checkDims(field string, dims []int) error {
	for i, d := range dims {
		if d <= 0 || d > maxDim {
			return fmt.Errorf("codex: %s[%d] must be in 1..%d, got %d", field, i, maxDim, d)
		}
	}
	return nil
}

func stack(dims []int, rng *rand.Rand) []dense {
	layers := make([]dense, len(dims)-1)
	for i := range layers {
		layers[i] = newDense(dims[i], dims[i+1], rng)
	}
	return layers
}

// newDense uses Xavier uniform init. rand.Float64 in [0,1) would make every
// weight positive and saturate the activations on the first forward pass.
func newDense(in, out int, rng *rand.Rand) dense {
	limit := math.Sqrt(6 / float64(in+out))
	w := make([]float64, in*out)
	for i := range w {
		w[i] = (rng.Float64()*2 - 1) * limit
	}
	return dense{w: w, b: make([]float64, out), in: in, out: out}
}

// Score returns one raw valuation per input row, in the same order. Valuations
// are unbounded; normalising them into allocations is the caller's job.
//
// Score never writes to the model or to the input, so one Net may be scored from
// many goroutines at once - but not while another goroutine is inside SetParams,
// which does write.
//
// Every feature must be finite. Checking here rather than on the output is not
// paranoia: math.Tanh(math.Inf(1)) is exactly 1, so under the default activation
// an infinite feature absorbs into a perfectly plausible finite valuation and
// the caller optimises against a fabricated stock forever.
func (c *Net) Score(stocks [][]float64) ([]float64, error) {
	for i, row := range stocks {
		if len(row) != c.features {
			return nil, fmt.Errorf("codex: stock %d has %d features, want %d", i, len(row), c.features)
		}
		for j, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("codex: stock %d feature %d is %v", i, j, v)
			}
		}
	}
	n := len(stocks)
	out := make([]float64, n)
	if n == 0 {
		return out, nil
	}

	width := 0
	c.each(func(d *dense) {
		if d.out > width {
			width = d.out
		}
	})
	embed := c.mix.in / 2
	embeddings := make([]float64, n*embed)
	front := make([]float64, width)
	back := make([]float64, width)

	// forward returns a slice of the buffer it wrote, which becomes the next
	// layer's input - so each call must write into the other buffer.
	for i, row := range stocks {
		src := row
		for j := range c.enc {
			src = c.enc[j].forward(front, src, c.act)
			front, back = back, front
		}
		copy(embeddings[i*embed:], src)
	}

	market := make([]float64, embed)
	for i := range n {
		for j, v := range embeddings[i*embed : (i+1)*embed] {
			market[j] += v
		}
	}
	for j := range market {
		market[j] /= float64(n)
	}

	// market is loop-invariant, so it goes into the upper half of mixIn once;
	// the loop below only ever overwrites the lower half.
	mixIn := make([]float64, 2*embed)
	copy(mixIn[embed:], market)
	last := len(c.head) - 1
	for i := range n {
		copy(mixIn, embeddings[i*embed:(i+1)*embed])
		src := c.mix.forward(front, mixIn, c.act)
		front, back = back, front
		for j := range c.head[:last] {
			src = c.head[j].forward(front, src, c.act)
			front, back = back, front
		}
		out[i] = c.head[last].forward(front, src, nil)[0]
		front, back = back, front
	}
	return out, nil
}

// Mutated returns a perturbed copy, adding Gaussian noise with standard
// deviation sigma to every parameter. The receiver is untouched, so it can be
// kept as the generation's elite and mutated repeatedly.
//
// rng must not be nil, and *rand.Rand is not safe for concurrent use: to mutate
// one parent from several goroutines, give each its own rng.
//
// sigma is a standard deviation, so a negative value behaves as its magnitude -
// Mutated cannot tell a sign bug in an annealing schedule from a deliberate one.
// A non-finite sigma panics: it would turn every parameter to NaN, and a model
// of NaNs still scores, still breeds, and still looks healthy to a caller.
func (c *Net) Mutated(sigma float64, rng *rand.Rand) *Net {
	if math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		panic(fmt.Sprintf("codex: sigma must be finite, got %v", sigma))
	}
	out := c.Clone()
	out.each(func(d *dense) {
		for i := range d.w {
			d.w[i] += rng.NormFloat64() * sigma
		}
		for i := range d.b {
			d.b[i] += rng.NormFloat64() * sigma
		}
	})
	return out
}

// Clone returns an independent copy sharing no memory with the receiver. Pair it
// with SetParams to load a stored parameter vector without re-running New.
func (c *Net) Clone() *Net {
	out := &Net{
		features: c.features,
		act:      c.act,
		enc:      slices.Clone(c.enc),
		mix:      c.mix,
		head:     slices.Clone(c.head),
	}
	out.each(func(d *dense) {
		d.w = slices.Clone(d.w)
		d.b = slices.Clone(d.b)
	})
	return out
}

// each visits every layer in a fixed order. Params and SetParams rely on that
// order being stable, so changing it invalidates every stored parameter vector.
func (c *Net) each(fn func(d *dense)) {
	for i := range c.enc {
		fn(&c.enc[i])
	}
	fn(&c.mix)
	for i := range c.head {
		fn(&c.head[i])
	}
}

// Params returns a copy of every weight and bias, in a stable order. Use it to
// persist a trained model, or to breed two of them by combining their slices.
func (c *Net) Params() []float64 {
	out := make([]float64, 0, c.ParamCount())
	c.each(func(d *dense) {
		out = append(out, d.w...)
		out = append(out, d.b...)
	})
	return out
}

// SetParams overwrites every weight and bias from p.
//
// It can only check that p is the right length, and different shapes can share
// one: a model with 213 parameters will happily load another's 213 and return
// plausible nonsense forever. Rebuild the target with New from the same Config
// the parameters were saved from.
//
// Parameters must be finite, for the same reason features must be: an infinite
// weight saturates through math.Tanh into a plausible, rankable, entirely
// fabricated valuation, and Params round-trips the corruption losslessly.
//
// SetParams writes in place, so it must not run while another goroutine is
// inside Score. It validates before writing, so a rejected call changes nothing.
func (c *Net) SetParams(p []float64) error {
	if len(c.enc) == 0 {
		return fmt.Errorf("codex: model was not built by New")
	}
	if n := c.ParamCount(); len(p) != n {
		return fmt.Errorf("codex: got %d parameters, want %d", len(p), n)
	}
	for i, v := range p {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("codex: parameter %d is %v", i, v)
		}
	}
	c.each(func(d *dense) {
		p = p[copy(d.w, p):]
		p = p[copy(d.b, p):]
	})
	return nil
}

// ParamCount returns the length Params returns and SetParams requires.
func (c *Net) ParamCount() int {
	n := 0
	c.each(func(d *dense) { n += len(d.w) + len(d.b) })
	return n
}
