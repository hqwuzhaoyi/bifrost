package bifrost

import (
	"context"
	"strings"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestIsErrorOnlyChunk verifies the predicate used by peekFirstStreamChunk to
// decide whether the first chunk represents an upstream error that should
// trigger fallback. Because no bytes have reached the caller at first-chunk
// peek time, any first chunk carrying BifrostError should be surfaced as an
// attempt failure even if a post-hook also attached an empty/auxiliary response
// object.
func TestIsErrorOnlyChunk(t *testing.T) {
	makeErr := func() *schemas.BifrostError {
		return &schemas.BifrostError{Error: &schemas.ErrorField{Message: "upstream overloaded"}}
	}

	cases := []struct {
		name string
		c    *schemas.BifrostStreamChunk
		want bool
	}{
		{
			name: "nil chunk",
			c:    nil,
			want: false,
		},
		{
			name: "no error",
			c:    &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{}},
			want: false,
		},
		{
			name: "error only — chat path",
			c:    &schemas.BifrostStreamChunk{BifrostError: makeErr()},
			want: true,
		},
		{
			name: "error + chat response object — first chunk still triggers fallback",
			c: &schemas.BifrostStreamChunk{
				BifrostError:        makeErr(),
				BifrostChatResponse: &schemas.BifrostChatResponse{},
			},
			want: true,
		},
		{
			name: "error + text completion response object — first chunk still triggers fallback",
			c: &schemas.BifrostStreamChunk{
				BifrostError:                  makeErr(),
				BifrostTextCompletionResponse: &schemas.BifrostTextCompletionResponse{},
			},
			want: true,
		},
		{
			name: "error + responses stream response object — first chunk still triggers fallback",
			c: &schemas.BifrostStreamChunk{
				BifrostError:                   makeErr(),
				BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{},
			},
			want: true,
		},
		{
			name: "error + passthrough response object — first chunk still triggers fallback",
			c: &schemas.BifrostStreamChunk{
				BifrostError:               makeErr(),
				BifrostPassthroughResponse: &schemas.BifrostPassthroughResponse{},
			},
			want: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := isErrorOnlyChunk(tc.c)
			if got != tc.want {
				t.Fatalf("isErrorOnlyChunk = %v, want %v", got, tc.want)
			}
		})
	}
}

// chatRequestFixture builds a minimal but well-formed BifrostRequest so
// PopulateExtraFields inside peekFirstStreamChunk has a non-nil sub-request to
// populate. The actual provider/model strings are passed separately to peek.
func chatRequestFixture() *schemas.BifrostRequest {
	return &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionStreamRequest,
		ChatRequest: &schemas.BifrostChatRequest{
			Provider: schemas.OpenAI,
			Model:    "gpt-test",
		},
	}
}

// TestPeekFirstStreamChunk_ErrorOnlyFirstChunk verifies the failure path:
// when the upstream's first chunk is an error-only chunk (the "fake 200"
// pattern), peek surfaces the embedded BifrostError so the outer fallback
// loop can engage. This is the entire point of the patch.
func TestPeekFirstStreamChunk_ErrorOnlyFirstChunk(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	stream := make(chan *schemas.BifrostStreamChunk, 1)
	stream <- &schemas.BifrostStreamChunk{
		BifrostError: &schemas.BifrostError{
			Error: &schemas.ErrorField{Message: "[1305] upstream overloaded"},
		},
	}
	close(stream)

	var b *Bifrost // peekFirstStreamChunk does not dereference the receiver
	req := chatRequestFixture()
	out, err := b.peekFirstStreamChunk(ctx, stream, req, "test-provider", "test-model")
	if out != nil {
		t.Fatalf("expected nil stream on error-only first chunk, got %v", out)
	}
	if err == nil {
		t.Fatalf("expected BifrostError surfaced, got nil")
	}
	if err.Error == nil || !strings.Contains(err.Error.Message, "1305") {
		t.Fatalf("expected original error message preserved, got %+v", err.Error)
	}
	if err.ExtraFields.OriginalModelRequested != "test-model" {
		t.Fatalf("expected PopulateExtraFields to set OriginalModelRequested=test-model, got %q", err.ExtraFields.OriginalModelRequested)
	}
}

// TestPeekFirstStreamChunk_RealFirstChunk verifies the happy path: when the
// upstream's first chunk carries real data, peek splices it back at the head
// of the stream and the consumer receives it in order alongside subsequent
// chunks. No data must be reordered or dropped.
func TestPeekFirstStreamChunk_RealFirstChunk(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	first := &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{ID: "chunk-1"}}
	second := &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{ID: "chunk-2"}}
	third := &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{ID: "chunk-3"}}

	stream := make(chan *schemas.BifrostStreamChunk, 3)
	stream <- first
	stream <- second
	stream <- third
	close(stream)

	var b *Bifrost
	req := chatRequestFixture()
	out, err := b.peekFirstStreamChunk(ctx, stream, req, "test-provider", "test-model")
	if err != nil {
		t.Fatalf("unexpected error on real first chunk: %+v", err)
	}
	if out == nil {
		t.Fatalf("expected non-nil spliced stream")
	}

	var got []string
	timeout := time.After(2 * time.Second)
	for {
		select {
		case chunk, ok := <-out:
			if !ok {
				goto done
			}
			if chunk != nil && chunk.BifrostChatResponse != nil {
				got = append(got, chunk.BifrostChatResponse.ID)
			}
		case <-timeout:
			t.Fatalf("timed out reading spliced stream; got so far: %v", got)
		}
	}
done:
	want := []string{"chunk-1", "chunk-2", "chunk-3"}
	if len(got) != len(want) {
		t.Fatalf("spliced stream length = %d, want %d (got=%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("spliced stream order mismatch at index %d: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}
}

// TestPeekFirstStreamChunk_ClosedBeforeAnyChunk verifies the empty-stream
// path: when the upstream channel closes without emitting any chunk at all,
// peek surfaces a synthetic connection-level error so the outer fallback loop
// can engage (otherwise the caller would silently get back an empty stream
// and produce a confusing zero-content response).
func TestPeekFirstStreamChunk_ClosedBeforeAnyChunk(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	stream := make(chan *schemas.BifrostStreamChunk)
	close(stream)

	var b *Bifrost
	req := chatRequestFixture()
	out, err := b.peekFirstStreamChunk(ctx, stream, req, "test-provider", "test-model")
	if out != nil {
		t.Fatalf("expected nil stream on closed-before-chunk path, got %v", out)
	}
	if err == nil {
		t.Fatalf("expected synthetic BifrostError, got nil")
	}
	if err.Error == nil || !strings.Contains(err.Error.Message, "without sending any chunk") {
		t.Fatalf("expected synthetic close-without-chunk error, got %+v", err.Error)
	}
	if err.ExtraFields.OriginalModelRequested != "test-model" {
		t.Fatalf("expected PopulateExtraFields to set OriginalModelRequested=test-model, got %q", err.ExtraFields.OriginalModelRequested)
	}
}

// TestPeekFirstStreamChunk_ContextCancelled verifies the cancellation path:
// when the caller's context is already cancelled by the time peek runs, peek
// returns a context-done error rather than blocking forever waiting for the
// first chunk. This guarantees the outer fallback loop is never stalled by a
// dead upstream that never emits.
func TestPeekFirstStreamChunk_ContextCancelled(t *testing.T) {
	innerCtx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel
	ctx := schemas.NewBifrostContext(innerCtx, schemas.NoDeadline)

	stream := make(chan *schemas.BifrostStreamChunk, 1)
	stream <- &schemas.BifrostStreamChunk{
		BifrostError: &schemas.BifrostError{Error: &schemas.ErrorField{Message: "should not reach peek"}},
	}
	close(stream)

	var b *Bifrost
	req := chatRequestFixture()
	out, err := b.peekFirstStreamChunk(ctx, stream, req, "test-provider", "test-model")
	if out != nil {
		t.Fatalf("expected nil stream on cancelled context, got %v", out)
	}
	if err == nil {
		t.Fatalf("expected context-done error, got nil")
	}
}

// TestFallbackExhaustedError_PrefersLastError verifies that when the entire
// fallback chain fails, the returned error is the *last* attempt's error (the
// terminal provider/model that exhausted the chain), not the primary error.
// This is the core invariant for the "all-fake-200 degraded" scenario: the
// client must see which provider actually failed last, not the first one.
func TestFallbackExhaustedError_PrefersLastError(t *testing.T) {
	primaryErr := &schemas.BifrostError{
		Error: &schemas.ErrorField{Message: "primary failed"},
	}
	primaryErr.ExtraFields.Provider = "provider-A"
	primaryErr.ExtraFields.OriginalModelRequested = "model-A"

	lastErr := &schemas.BifrostError{
		Error: &schemas.ErrorField{Message: "last fallback failed"},
	}
	lastErr.ExtraFields.Provider = "provider-Z"
	lastErr.ExtraFields.OriginalModelRequested = "model-Z"

	got := fallbackExhaustedError(primaryErr, lastErr)
	if got != lastErr {
		t.Fatalf("expected lastErr returned, got different error: %+v", got)
	}
}

// TestFallbackExhaustedError_NilLastFallsBackToPrimary verifies the defensive
// path: if lastErr is nil (all fallbacks were skipped, none executed), the
// primary error is returned as the only available signal.
func TestFallbackExhaustedError_NilLastFallsBackToPrimary(t *testing.T) {
	primaryErr := &schemas.BifrostError{
		Error: &schemas.ErrorField{Message: "primary failed"},
	}

	got := fallbackExhaustedError(primaryErr, nil)
	if got != primaryErr {
		t.Fatalf("expected primaryErr returned when lastErr is nil, got: %+v", got)
	}
}
