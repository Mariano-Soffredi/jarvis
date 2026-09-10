package codex

import "testing"

func benchScore(b *testing.B, act Activation) {
	cfg := Config{FeaturesPerStock: 8, Encoder: []int{16, 16}, Mix: 16, Head: []int{16}, Hidden: act}
	m, err := New(cfg, seeded(1))
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	in := stocks(200, 8)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Score(in); err != nil {
			b.Fatal(err)
		}
	}
}

// The pair is the point: it measures what the safe default costs against the
// fast one, which is the tradeoff ReLU's doc comment claims.
func BenchmarkScoreTanh(b *testing.B) { benchScore(b, nil) }
func BenchmarkScoreReLU(b *testing.B) { benchScore(b, ReLU) }
