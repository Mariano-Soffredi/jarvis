package codex

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func testConfig() Config {
	return Config{
		FeaturesPerStock: 4,
		Encoder:          []int{8, 6},
		Mix:              6,
		Head:             []int{5},
	}
}

func seeded(n uint64) *rand.Rand { return rand.New(rand.NewPCG(n, 1)) }

func newNet(t *testing.T, cfg Config, rng *rand.Rand) *Net {
	t.Helper()
	m, err := New(cfg, rng)
	if err != nil {
		t.Fatalf("New(%+v): %v", cfg, err)
	}
	return m
}

func mustScore(t *testing.T, m *Net, in [][]float64) []float64 {
	t.Helper()
	out, err := m.Score(in)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	return out
}

func stocks(n, k int) [][]float64 {
	in := make([][]float64, n)
	for i := range in {
		in[i] = make([]float64, k)
		for j := range in[i] {
			in[i][j] = float64(i*k+j) / 10
		}
	}
	return in
}

// TestForwardPassAppliesHiddenActivation is the only test that can tell an
// activated final head layer from a linear one, since under an identity
// activation the two are indistinguishable. Square keeps it hand-computable:
//
//	e0 = (2*1+0.5)^2 = 6.25      e1 = (2*3+0.5)^2 = 42.25
//	market = (6.25+42.25)/2 = 24.25
//	mix0 = (6.25 + 3*24.25)^2 = 79^2 = 6241
//	mix1 = (42.25 + 3*24.25)^2 = 115^2 = 13225
//	out0 = 10*6241 - 1 = 62409   out1 = 10*13225 - 1 = 132249
func TestForwardPassAppliesHiddenActivation(t *testing.T) {
	square := func(x float64) float64 { return x * x }
	m := newNet(t, Config{FeaturesPerStock: 1, Encoder: []int{1}, Mix: 1, Hidden: square}, seeded(1))

	// each()'s order: enc.w enc.b mix.w mix.b head.w head.b
	if err := m.SetParams([]float64{2, 0.5, 1, 3, 0, 10, -1}); err != nil {
		t.Fatalf("SetParams: %v", err)
	}

	got := mustScore(t, m, [][]float64{{1}, {3}})
	want := []float64{62409, 132249}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestForwardPassWithWidthAndDepth pins weight-row indexing and the double
// buffers, which a 1-wide net cannot exercise: every layer here is 2 wide and
// the encoder has two of them. Identity activation, by hand for [1,2] and [3,1]:
//
//	enc0: [1,2]->[1*1+0*2+0, 0*1+2*2+1] = [1,5]   [3,1]->[3,3]
//	enc1: [1,5]->[1+5, 2*1+0]           = [6,2]   [3,3]->[6,6]
//	market = [(6+6)/2, (2+6)/2] = [6,4]
//	mixA = [6,2,6,4] -> [1*6+3*2, 1*6+1*4] = [12,10]
//	mixB = [6,6,6,4] -> [6+18, 6+4]        = [24,10]
//	head0: [12,10]->[12+10, 2*12] = [22,24]   [24,10]->[34,48]
//	head1: [22,24]->[24, 22+24]   = [24,46]   [34,48]->[48,82]
//	outA = 10*24 + 46 - 5 = 281   outB = 10*48 + 82 - 5 = 557
//
// The head must be two layers wide AND two layers deep for aliasing to bite: a
// layer with out==1 reads every input before its single write, and an identity
// weight matrix is invariant under overwriting, so either would hide the bug.
func TestForwardPassWithWidthAndDepth(t *testing.T) {
	identity := func(x float64) float64 { return x }
	cfg := Config{FeaturesPerStock: 2, Encoder: []int{2, 2}, Mix: 2, Head: []int{2, 2}, Hidden: identity}
	m := newNet(t, cfg, seeded(1))

	params := []float64{
		1, 0, 0, 2, // enc0.w
		0, 1, // enc0.b
		1, 1, 2, 0, // enc1.w
		0, 0, // enc1.b
		1, 3, 0, 0, 0, 0, 1, 1, // mix.w
		0, 0, // mix.b
		1, 1, 2, 0, // head0.w
		0, 0, // head0.b
		0, 1, 1, 1, // head1.w
		0, 0, // head1.b
		10, 1, // head2.w
		-5, // head2.b
	}
	if err := m.SetParams(params); err != nil {
		t.Fatalf("SetParams: %v", err)
	}

	got := mustScore(t, m, [][]float64{{1, 2}, {3, 1}})
	want := []float64{281, 557}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// The market average is the only channel between items, so scoring the same
// stock beside different peers must move its valuation.
func TestValuationDependsOnTheRestOfTheMarket(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	subject := []float64{1, 2, 3, 4}

	calm := mustScore(t, m, [][]float64{subject, {0, 0, 0, 0}})
	wild := mustScore(t, m, [][]float64{subject, {9, -9, 9, -9}})

	if calm[0] == wild[0] {
		t.Errorf("valuation %v ignored the other stock", calm[0])
	}
}

// Reordering the input must reorder the output and nothing else. Equality holds
// only to floating-point precision: the market average accumulates in input
// order, so the last bit can move and exact ties may re-sort differently.
func TestScoreIsPermutationEquivariant(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	a, b, c := []float64{1, 2, 3, 4}, []float64{5, 6, 7, 8}, []float64{9, 1, 2, 3}

	fwd := mustScore(t, m, [][]float64{a, b, c})
	rev := mustScore(t, m, [][]float64{c, b, a})

	for i, j := range []int{2, 1, 0} {
		if math.Abs(fwd[i]-rev[j]) > 1e-12 {
			t.Errorf("reordering changed valuations: %v vs reversed %v", fwd, rev)
		}
	}
}

// A model whose valuations are all equal cannot rank anything, which is the
// whole job. Guard the default config against silently producing one.
func TestDefaultConfigProducesUsableValuations(t *testing.T) {
	in := stocks(12, 4)
	for seed := uint64(0); seed < 50; seed++ {
		out := mustScore(t, newNet(t, testConfig(), seeded(seed)), in)

		spread := false
		for _, v := range out {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("seed %d: non-finite valuation %v", seed, v)
			}
			if v != out[0] {
				spread = true
			}
		}
		if !spread {
			t.Fatalf("seed %d: every stock valued %v; model cannot rank", seed, out[0])
		}
	}
}

// Params must straddle zero. An all-positive init - what the obvious
// rand.Float64 gives - biases every layer the same way and saturates a bounded
// activation, and evolution has no gradient to pull it back.
func TestInitIsZeroCentered(t *testing.T) {
	p := newNet(t, testConfig(), seeded(1)).Params()

	pos, neg, sum := 0, 0, 0.0
	for _, v := range p {
		switch {
		case v > 0:
			pos++
		case v < 0:
			neg++
		}
		sum += v
	}
	if pos == 0 || neg == 0 {
		t.Fatalf("init produced %d positive and %d negative params; want both signs", pos, neg)
	}
	if mean := math.Abs(sum / float64(len(p))); mean > 0.05 {
		t.Errorf("init mean is %v, want near zero", mean)
	}
}

func TestOneModelScoresAnyStockCount(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	want := m.ParamCount()

	for _, n := range []int{1, 20, 500} {
		if out := mustScore(t, m, stocks(n, 4)); len(out) != n {
			t.Errorf("Score(%d) returned %d valuations", n, len(out))
		}
	}
	if got := m.ParamCount(); got != want {
		t.Errorf("parameter count moved with the stock count: %d, want %d", got, want)
	}
}

func TestMutatedCopiesAndPerturbsEveryParam(t *testing.T) {
	parent := newNet(t, testConfig(), seeded(1))
	before := parent.Params()

	child := parent.Mutated(0.1, seeded(9))

	if !slices.Equal(before, parent.Params()) {
		t.Error("Mutated changed the parent")
	}
	after := child.Params()
	for i := range before {
		if before[i] == after[i] {
			t.Fatalf("param %d of %d unchanged; Mutated skipped part of the model", i, len(before))
		}
	}
}

// A population drawn from one rng must diverge; separate rngs seeded alike must
// not. This is the difference between a population and a monoculture.
func TestPopulationFromOneRngDiverges(t *testing.T) {
	rng := seeded(1)
	cfg := testConfig()
	a, b := newNet(t, cfg, rng), newNet(t, cfg, rng)
	if slices.Equal(a.Params(), b.Params()) {
		t.Error("two models from one rng are identical")
	}

	c, d := newNet(t, cfg, seeded(7)), newNet(t, cfg, seeded(7))
	if !slices.Equal(c.Params(), d.Params()) {
		t.Error("equally seeded rngs produced different models")
	}
}

// The save/restore path a caller uses to persist an elite. The model must be
// genuinely perturbed in between, or this proves nothing about SetParams.
func TestParamsRoundTrip(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	in := stocks(5, 4)
	before := mustScore(t, m, in)
	saved := m.Params()

	if err := m.SetParams(m.Mutated(0.5, seeded(2)).Params()); err != nil {
		t.Fatalf("SetParams(mutated): %v", err)
	}
	if mutated := mustScore(t, m, in); slices.Equal(before, mutated) {
		t.Fatal("model was not perturbed; the rest of this test would be vacuous")
	}

	if err := m.SetParams(saved); err != nil {
		t.Fatalf("SetParams(saved): %v", err)
	}
	if after := mustScore(t, m, in); !slices.Equal(before, after) {
		t.Errorf("round trip changed output: %v -> %v", before, after)
	}
}

func TestParamsReturnsACopy(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	before := m.Params()

	p := m.Params()
	for i := range p {
		p[i] = 99
	}

	if !slices.Equal(before, m.Params()) {
		t.Error("mutating the slice from Params changed the model")
	}
}

func TestSetParamsRejectsWrongLength(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	want := m.ParamCount()

	for _, p := range [][]float64{make([]float64, want-1), make([]float64, want+1), nil} {
		if err := m.SetParams(p); err == nil {
			t.Errorf("SetParams accepted %d params, want %d", len(p), want)
		}
	}
}

// An infinite parameter saturates through math.Tanh into a finite, ordered,
// entirely fabricated valuation - so it must be caught where it enters, not by
// inspecting the output.
func TestSetParamsRejectsNonFiniteParams(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		p := m.Params()
		p[len(p)/2] = bad
		if err := m.SetParams(p); err == nil {
			t.Errorf("SetParams accepted %v as a parameter", bad)
		}
	}
	if err := (&Net{}).SetParams(nil); err == nil {
		t.Error("SetParams on a zero Net returned no error")
	}
}

// A NaN sigma - a divide by zero in an annealing schedule - would turn every
// parameter to NaN and still produce a model that scores, breeds and looks fine.
func TestMutatedRejectsNonFiniteSigma(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	for _, bad := range []float64{math.NaN(), math.Inf(1)} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Mutated accepted sigma %v", bad)
				}
			}()
			m.Mutated(bad, seeded(1))
		}()
	}
}

// One bad tick must not silently poison the whole portfolio through the market
// average. Inf matters as much as NaN: math.Tanh(+Inf) is exactly 1, so the
// default activation would absorb it into a plausible finite valuation.
func TestScoreRejectsNonFiniteFeatures(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		in := stocks(5, 4)
		in[3][2] = bad
		if _, err := m.Score(in); err == nil {
			t.Errorf("Score accepted %v as a feature", bad)
		}
	}
}

func TestScoreRejectsWrongFeatureCount(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	if _, err := m.Score([][]float64{{1, 2, 3, 4}, {1, 2}}); err == nil {
		t.Error("Score accepted a row with the wrong feature count")
	}
}

func TestScoreAcceptsNoStocks(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	out, err := m.Score(nil)
	if err != nil {
		t.Fatalf("Score(nil): %v", err)
	}
	if out == nil || len(out) != 0 {
		t.Errorf("Score(nil) = %v, want a non-nil empty slice", out)
	}
}

func TestNewRejectsBadConfig(t *testing.T) {
	tests := map[string]func(*Config){
		"no features":     func(c *Config) { c.FeaturesPerStock = 0 },
		"overflow dims":   func(c *Config) { c.FeaturesPerStock = 1 << 62 },
		"huge encoder":    func(c *Config) { c.Encoder = []int{1 << 62} },
		"no encoder":      func(c *Config) { c.Encoder = nil },
		"zero-width enc":  func(c *Config) { c.Encoder = []int{8, 0} },
		"no mix":          func(c *Config) { c.Mix = 0 },
		"zero-width head": func(c *Config) { c.Head = []int{0} },
	}
	for name, corrupt := range tests {
		cfg := testConfig()
		corrupt(&cfg)
		if _, err := New(cfg, seeded(1)); err == nil {
			t.Errorf("%s: New accepted an invalid config", name)
		}
	}
	if _, err := New(testConfig(), nil); err == nil {
		t.Error("New accepted a nil rng")
	}
}

func TestConcurrentScoreIsSafe(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	in := stocks(50, 4)
	want := mustScore(t, m, in)

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				got, err := m.Score(in)
				if err != nil {
					t.Errorf("concurrent Score: %v", err)
					return
				}
				if !slices.Equal(got, want) {
					t.Errorf("concurrent Score diverged: %v", got)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// One generation of the loop this package exists to serve. Compiled, so it
// cannot drift out of sync with the API the way a doc-comment sketch would.
func Example() {
	rng := rand.New(rand.NewPCG(42, 1))
	cfg := Config{FeaturesPerStock: 3, Encoder: []int{8, 6}, Mix: 6, Head: []int{4}}

	population := make([]*Net, 20)
	for i := range population {
		m, err := New(cfg, rng)
		if err != nil {
			panic(err)
		}
		population[i] = m
	}

	today := [][]float64{{101.2, 0.4, 55}, {88.7, 0.1, 41}, {19.3, 0.9, 72}}

	best, bestScore := population[0], math.Inf(-1)
	for _, m := range population {
		valuations, err := m.Score(today)
		if err != nil {
			panic(err)
		}
		if fitness := valuations[0]; fitness > bestScore {
			best, bestScore = m, fitness
		}
	}

	for i := range population {
		population[i] = best.Mutated(0.05, rng)
	}

	valuations, _ := best.Score(today)
	fmt.Println(len(valuations), best.ParamCount())
	// Output: 3 197
}
