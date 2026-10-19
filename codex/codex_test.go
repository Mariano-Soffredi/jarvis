package codex

import (
	"crypto/sha256"
	"encoding/binary"
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
	out, _, err := m.Score(in, nil)
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
		if _, _, err := m.Score(in, nil); err == nil {
			t.Errorf("Score accepted %v as a feature", bad)
		}
	}
}

func TestScoreRejectsWrongFeatureCount(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	if _, _, err := m.Score([][]float64{{1, 2, 3, 4}, {1, 2}}, nil); err == nil {
		t.Error("Score accepted a row with the wrong feature count")
	}
}

func TestScoreAcceptsNoStocks(t *testing.T) {
	m := newNet(t, testConfig(), seeded(1))
	out, _, err := m.Score(nil, nil)
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
		"negative ctx":    func(c *Config) { c.Context = -1 },
		"huge ctx":        func(c *Config) { c.Context = 1 << 62 },
		"zero-width pool": func(c *Config) { c.Pool = []int{0} },
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
				got, _, err := m.Score(in, nil)
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
		valuations, _, err := m.Score(today, nil)
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

	valuations, _, _ := best.Score(today, nil)
	fmt.Println(len(valuations), best.ParamCount())
	// Output: 3 197
}

func digest(vs ...[]float64) string {
	h := sha256.New()
	for _, v := range vs {
		for _, x := range v {
			binary.Write(h, binary.LittleEndian, math.Float64bits(x))
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// The digest was taken from v0.0.0-20260911150921-dbd85ce4bb38 at strategy's
// default shape. It pins init draw order, Params order, Mutated's draw order
// and Score's bits, which is what keeps stored champions loadable.
func TestNoContextNoPoolMatchesThePreviousRelease(t *testing.T) {
	m := newNet(t, Config{FeaturesPerStock: 41, Encoder: []int{8, 8}, Mix: 8}, seeded(3))
	child := m.Mutated(0.1, seeded(4))
	in := stocks(30, 41)

	a, pa, err := m.Score(in, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := mustScore(t, child, in)

	if got := m.ParamCount(); got != 553 {
		t.Errorf("ParamCount = %d, want 553", got)
	}
	if pa != 0 {
		t.Errorf("pool output without a pool = %v, want 0", pa)
	}
	const want = "1c39f70d6eddec6e96b5b99f4454458c06899e5541ff7fbb9533cf3b05624dd5"
	if got := digest(m.Params(), child.Params(), a, b); got != want {
		t.Errorf("digest = %s, want %s", got, want)
	}
}

func TestContextAddsMixTimesContextParams(t *testing.T) {
	cfg := testConfig()
	base := newNet(t, cfg, seeded(1)).ParamCount()
	cfg.Context = 3
	if got, want := newNet(t, cfg, seeded(1)).ParamCount(), base+cfg.Mix*3; got != want {
		t.Errorf("ParamCount = %d, want %d", got, want)
	}
}

func TestValuationAndPoolDependOnContext(t *testing.T) {
	cfg := testConfig()
	cfg.Context, cfg.Pool = 2, []int{}
	m := newNet(t, cfg, seeded(1))
	in := stocks(4, 4)

	calm, pc, err := m.Score(in, []float64{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	wild, pw, err := m.Score(in, []float64{3, -3})
	if err != nil {
		t.Fatal(err)
	}
	if calm[0] == wild[0] {
		t.Errorf("valuation %v ignored the context", calm[0])
	}
	if pc == pw {
		t.Errorf("pool output %v ignored the context", pc)
	}
}

func TestScoreRejectsBadContext(t *testing.T) {
	cfg := testConfig()
	cfg.Context = 2
	m := newNet(t, cfg, seeded(1))
	for _, ctx := range [][]float64{nil, {1}, {1, 2, 3}, {1, math.NaN()}, {math.Inf(1), 0}, {0, math.Inf(-1)}} {
		if _, _, err := m.Score(stocks(3, 4), ctx); err == nil {
			t.Errorf("Score accepted context %v", ctx)
		}
	}
	if _, _, err := newNet(t, testConfig(), seeded(1)).Score(stocks(3, 4), []float64{1}); err == nil {
		t.Error("a net without context accepted one")
	}
}

// Identity activation, by hand for rows [1] and [3] with ctx [0.5]:
//
//	e = 2x+0.5: e0 = 2.5, e1 = 6.5, market = 4.5
//	mix = e + 3*market + 4*ctx: mix0 = 18, mix1 = 22
//	out = 10*mix - 1: 179, 219
//	pool = 2*market + 5*ctx + 1 = 12.5
func TestForwardPassWithContextAndPool(t *testing.T) {
	identity := func(x float64) float64 { return x }
	cfg := Config{FeaturesPerStock: 1, Encoder: []int{1}, Mix: 1, Context: 1, Pool: []int{}, Hidden: identity}
	m := newNet(t, cfg, seeded(1))

	// each()'s order: enc, mix (columns e, market, ctx), head, pool (columns market, ctx)
	if err := m.SetParams([]float64{2, 0.5, 1, 3, 4, 0, 10, -1, 2, 5, 1}); err != nil {
		t.Fatalf("SetParams: %v", err)
	}

	got, pool, err := m.Score([][]float64{{1}, {3}}, []float64{0.5})
	if err != nil {
		t.Fatal(err)
	}
	if want := []float64{179, 219}; !slices.Equal(got, want) {
		t.Errorf("valuations %v, want %v", got, want)
	}
	if pool != 12.5 {
		t.Errorf("pool %v, want 12.5", pool)
	}
}

// Appending the pool last in each() and in New's draws is what lets a stored
// champion be padded into a net with a fresh pooled head.
func TestPoolIsAppendedLast(t *testing.T) {
	cfg := testConfig()
	cfg.Context = 2
	plain := newNet(t, cfg, seeded(5))
	cfg.Pool = []int{3}
	pooled := newNet(t, cfg, seeded(5))

	embed := cfg.Encoder[len(cfg.Encoder)-1]
	extra := (embed+2)*3 + 3 + 3 + 1
	if got, want := pooled.ParamCount(), plain.ParamCount()+extra; got != want {
		t.Fatalf("ParamCount = %d, want %d", got, want)
	}
	if !slices.Equal(pooled.Params()[:plain.ParamCount()], plain.Params()) {
		t.Error("the pooled net's params do not start with the plain net's")
	}

	in, ctx := stocks(5, 4), []float64{1, -1}
	a, _, _ := plain.Score(in, ctx)
	b, _, _ := pooled.Score(in, ctx)
	if !slices.Equal(a, b) {
		t.Errorf("adding a pool changed the valuations: %v vs %v", a, b)
	}
}

func TestCloneAndMutatedCarryContextAndPool(t *testing.T) {
	cfg := testConfig()
	cfg.Context, cfg.Pool = 2, []int{3}
	m := newNet(t, cfg, seeded(1))
	in, ctx := stocks(5, 4), []float64{1, -1}
	want, wantPool, _ := m.Score(in, ctx)

	clone := m.Clone()
	got, gotPool, err := clone.Score(in, ctx)
	if err != nil {
		t.Fatalf("clone Score: %v", err)
	}
	if !slices.Equal(got, want) || gotPool != wantPool {
		t.Errorf("clone scored %v %v, want %v %v", got, gotPool, want, wantPool)
	}
	cp := clone.Params()
	cp[len(cp)-1] += 1
	if err := clone.SetParams(cp); err != nil {
		t.Fatal(err)
	}
	if _, p, _ := m.Score(in, ctx); p != wantPool {
		t.Error("changing the clone's pool changed the original")
	}

	before := m.Params()
	after := m.Mutated(0.1, seeded(9)).Params()
	if len(after) != len(before) {
		t.Fatalf("Mutated has %d params, want %d", len(after), len(before))
	}
	for i := range before {
		if before[i] == after[i] {
			t.Fatalf("param %d of %d unchanged by Mutated", i, len(before))
		}
	}
}
