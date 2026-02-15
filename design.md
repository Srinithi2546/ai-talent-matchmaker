# Design Document: Semantic Similarity Scoring Engine

## Overview

The Semantic Similarity Scoring Engine is a core AI module that compares resumes and job descriptions using embeddings and cosine similarity. The engine converts text documents into high-dimensional vector representations (embeddings) using pre-trained language models, then computes cosine similarity between these vectors to generate match scores ranging from 0.0 to 1.0.

The design emphasizes:
- **Accuracy**: Using state-of-the-art embedding models for semantic understanding
- **Performance**: Caching embeddings and supporting batch processing
- **Reliability**: Robust error handling and validation
- **Flexibility**: Configurable embedding models to balance accuracy and performance

## Architecture

The system follows a layered architecture:

```
┌─────────────────────────────────────────┐
│         API Layer                       │
│  (ScoringEngine public interface)      │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Business Logic Layer               │
│  - Similarity computation               │
│  - Score interpretation                 │
│  - Batch processing orchestration       │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Embedding Layer                    │
│  - Text preprocessing                   │
│  - Embedding generation                 │
│  - Embedding normalization              │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Infrastructure Layer               │
│  - Embedding cache                      │
│  - Model client (API/local)             │
│  - Serialization                        │
└─────────────────────────────────────────┘
```

### Key Design Decisions

1. **Embedding Model Choice**: Support for multiple embedding models (OpenAI, Sentence-Transformers, etc.) with a unified interface
2. **Caching Strategy**: LRU cache for embeddings with configurable size to balance memory and performance
3. **Batch Processing**: Vectorized operations for computing multiple similarities efficiently
4. **Error Handling**: Fail-fast validation with descriptive errors, retry logic for transient failures

## Components and Interfaces

### ScoringEngine (Main Interface)

The primary interface for all scoring operations.

```
class ScoringEngine:
    def __init__(config: EngineConfig):
        # Initialize with configuration (model type, cache size, etc.)
        pass
    
    def embed_text(text: str) -> Embedding:
        # Convert text to embedding vector
        # Validates input, handles chunking/truncation
        # Returns normalized embedding
        pass
    
    def compute_similarity(resume_embedding: Embedding, 
                          job_embedding: Embedding) -> float:
        # Compute cosine similarity between embeddings
        # Returns score between 0.0 and 1.0
        pass
    
    def score_match(resume_text: str, job_text: str) -> ScoringResult:
        # End-to-end: embed both texts and compute similarity
        # Returns result with score and metadata
        pass
    
    def score_batch(resume_texts: List[str], 
                   job_text: str) -> List[ScoringResult]:
        # Batch process multiple resumes against one job
        # Optimizes by computing job embedding once
        pass
    
    def interpret_score(score: float) -> ScoreInterpretation:
        # Convert numerical score to human-readable interpretation
        pass
```

### EmbeddingGenerator

Handles text-to-embedding conversion with model abstraction.

```
interface EmbeddingModel:
    def generate_embedding(text: str) -> Vector:
        # Model-specific embedding generation
        pass
    
    def get_dimension() -> int:
        # Return embedding dimension for this model
        pass
    
    def get_max_tokens() -> int:
        # Return maximum token limit
        pass

class EmbeddingGenerator:
    def __init__(model: EmbeddingModel, cache: EmbeddingCache):
        pass
    
    def generate(text: str) -> Embedding:
        # Check cache first
        # Preprocess text (normalize whitespace, handle length)
        # Generate embedding via model
        # Normalize embedding vector
        # Cache result
        # Return Embedding object
        pass
```

### SimilarityComputer

Computes cosine similarity with validation.

```
class SimilarityComputer:
    @staticmethod
    def cosine_similarity(embedding1: Embedding, 
                         embedding2: Embedding) -> float:
        # Validate dimensions match
        # Compute dot product of normalized vectors
        # Return similarity score (0.0 to 1.0)
        pass
    
    @staticmethod
    def batch_similarity(embeddings: List[Embedding], 
                        reference: Embedding) -> List[float]:
        # Vectorized computation for efficiency
        # Validate all dimensions match
        # Return list of similarity scores
        pass
```

### EmbeddingCache

LRU cache for storing computed embeddings.

```
class EmbeddingCache:
    def __init__(max_size: int):
        pass
    
    def get(text_hash: str) -> Optional[Embedding]:
        # Retrieve cached embedding if exists
        pass
    
    def put(text_hash: str, embedding: Embedding):
        # Store embedding with LRU eviction
        pass
    
    def clear():
        # Clear all cached embeddings
        pass
    
    def get_stats() -> CacheStats:
        # Return hit rate, size, etc.
        pass
```

## Data Models

### Embedding

```
class Embedding:
    vector: List[float]      # The embedding vector
    dimension: int           # Vector dimension
    model_name: str          # Model used to generate
    normalized: bool         # Whether vector is normalized
    
    def to_json() -> str:
        # Serialize to JSON
        pass
    
    @staticmethod
    def from_json(json_str: str) -> Embedding:
        # Deserialize from JSON
        pass
    
    def magnitude() -> float:
        # Compute L2 norm
        pass
    
    def normalize() -> Embedding:
        # Return normalized copy
        pass
```

### ScoringResult

```
class ScoringResult:
    score: float                    # Similarity score (0.0-1.0)
    interpretation: str             # Human-readable interpretation
    resume_embedding: Embedding     # Resume embedding (optional)
    job_embedding: Embedding        # Job embedding (optional)
    processing_time_ms: float       # Time taken
    cache_hit: bool                 # Whether cache was used
    
    def to_json() -> str:
        pass
```

### EngineConfig

```
class EngineConfig:
    model_type: str                 # "openai", "sentence-transformers", etc.
    model_name: str                 # Specific model identifier
    cache_size: int                 # Max cached embeddings
    max_retries: int                # Retry attempts for failures
    retry_delay_ms: int             # Initial retry delay
    score_thresholds: Dict[str, Tuple[float, float]]  # Interpretation ranges
    
    @staticmethod
    def default() -> EngineConfig:
        # Return sensible defaults
        pass
```

### Score Interpretation Ranges

Default score interpretation:
- **Excellent Match** (0.85 - 1.0): Strong semantic alignment
- **Good Match** (0.70 - 0.84): Significant overlap in qualifications
- **Fair Match** (0.50 - 0.69): Some relevant qualifications
- **Poor Match** (0.0 - 0.49): Limited alignment

## Correctness Properties


A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Valid text produces embeddings

*For any* valid non-empty text input, the embedding generation should produce a numerical vector with consistent dimension matching the configured model.

**Validates: Requirements 1.1, 1.2**

### Property 2: Whitespace-only text is rejected

*For any* string composed entirely of whitespace characters (spaces, tabs, newlines), the embedding generation should return an error indicating invalid input.

**Validates: Requirements 1.3**

### Property 3: Long text is handled gracefully

*For any* text exceeding the model's token limit, the embedding generation should complete successfully (via chunking or truncation) and return a valid embedding with appropriate warnings.

**Validates: Requirements 1.4**

### Property 4: Model consistency within engine instance

*For any* sequence of embedding operations on the same engine instance, all generated embeddings should have the same dimension and model identifier.

**Validates: Requirements 1.5**

### Property 5: Cosine similarity bounds

*For any* pair of valid embeddings with matching dimensions, the computed similarity score should be between 0.0 and 1.0 inclusive.

**Validates: Requirements 2.1, 2.2**

### Property 6: Dimension mismatch detection

*For any* pair of embeddings with different dimensions, attempting to compute similarity should return an error indicating incompatible embeddings.

**Validates: Requirements 2.3**

### Property 7: Null input detection

*For any* similarity computation where at least one embedding is null or invalid, the operation should return an error indicating missing input.

**Validates: Requirements 2.4**

### Property 8: Normalization equivalence

*For any* pair of embeddings, computing similarity on unnormalized embeddings should produce the same result as normalizing them first then computing similarity.

**Validates: Requirements 2.5**

### Property 9: Batch result count matches input count

*For any* batch of resume texts and a job description, the number of similarity scores returned should equal the number of resume texts provided.

**Validates: Requirements 3.1**

### Property 10: Batch order preservation

*For any* batch of resume texts processed against a job description, the order of results should match the order of input resumes.

**Validates: Requirements 3.2**

### Property 11: Batch partial failure handling

*For any* batch containing at least one invalid resume text, the valid resumes should still be processed and the invalid ones should be reported as failures without stopping the entire batch.

**Validates: Requirements 3.3**

### Property 12: Batch embedding efficiency

*For any* batch processing operation, the job description embedding should be computed exactly once regardless of the number of resumes in the batch.

**Validates: Requirements 3.4**

### Property 13: Score interpretation mapping

*For any* similarity score between 0.0 and 1.0, the interpretation function should return a category that falls within the defined score ranges (excellent, good, fair, or poor).

**Validates: Requirements 4.1, 4.2**

### Property 14: Model configuration acceptance

*For any* supported embedding model identifier, initializing the engine with that model should succeed and subsequent embeddings should use that model.

**Validates: Requirements 5.1**

### Property 15: Unsupported model rejection

*For any* unsupported or invalid model identifier, initializing the engine should return an error that includes a list of supported models.

**Validates: Requirements 5.2**

### Property 16: Cross-model comparison prevention

*For any* two embeddings generated by different embedding models, attempting to compute their similarity should return an error indicating incompatible models.

**Validates: Requirements 5.4**

### Property 17: Embedding cache effectiveness

*For any* text that is embedded multiple times, the second and subsequent embedding operations should return cached results (verifiable through cache hit metrics).

**Validates: Requirements 6.1, 6.3**

### Property 18: Cache metrics tracking

*For any* sequence of embedding operations, the cache should maintain accurate metrics including hit count, miss count, and hit rate.

**Validates: Requirements 6.4**

### Property 19: Descriptive error messages

*For any* invalid input (empty text, null values, mismatched dimensions), the error message should include specific information about what was invalid and why.

**Validates: Requirements 7.1**

### Property 20: Retry with exponential backoff

*For any* transient network error during model calls, the system should retry up to the configured maximum with exponentially increasing delays between attempts.

**Validates: Requirements 7.3**

### Property 21: Error logging completeness

*For any* error that occurs during processing, the system should log the error with context including input parameters, operation type, and timestamp.

**Validates: Requirements 7.4**

### Property 22: Serialization precision preservation

*For any* embedding with floating-point values, serializing to JSON and checking the serialized values should show precision is maintained to at least 6 decimal places.

**Validates: Requirements 8.1**

### Property 23: Deserialization validation

*For any* malformed or invalid JSON string, attempting to deserialize it as an embedding should return an error indicating the specific validation failure.

**Validates: Requirements 8.2**

### Property 24: JSON serialization support

*For any* valid embedding or scoring result, serializing to JSON should produce valid JSON that can be parsed by standard JSON parsers.

**Validates: Requirements 8.3**

### Property 25: Serialization round-trip

*For any* valid embedding, serializing to JSON then deserializing should produce an embedding with equivalent vector values (within floating-point precision tolerance).

**Validates: Requirements 8.4**

## Error Handling

### Error Categories

1. **Validation Errors**: Invalid input, empty text, dimension mismatches
   - Return immediately with descriptive error
   - No retries
   - HTTP 400 equivalent

2. **Model Errors**: Model unavailable, API failures, token limit exceeded
   - Retry with exponential backoff for transient errors
   - Return error after max retries
   - HTTP 503 equivalent

3. **Configuration Errors**: Invalid model specified, missing API keys
   - Fail at initialization time
   - Provide clear guidance on fixing configuration
   - HTTP 500 equivalent

### Error Response Format

```
class ScoringError:
    error_type: str          # "validation", "model", "configuration"
    message: str             # Human-readable description
    details: Dict[str, Any]  # Additional context
    timestamp: datetime
    retry_after: Optional[int]  # Seconds to wait before retry (if applicable)
```

### Retry Strategy

- **Max Retries**: 3 (configurable)
- **Initial Delay**: 100ms
- **Backoff Multiplier**: 2x
- **Max Delay**: 5 seconds
- **Retryable Errors**: Network timeouts, rate limits, temporary service unavailability
- **Non-Retryable Errors**: Authentication failures, invalid input, unsupported operations

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests to ensure comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs using randomized testing

Both approaches are complementary and necessary. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing Configuration

We will use a property-based testing library appropriate for the implementation language:
- **Python**: Hypothesis
- **TypeScript/JavaScript**: fast-check
- **Java**: jqwik
- **Go**: gopter

**Configuration Requirements**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `Feature: semantic-similarity-scoring-engine, Property {number}: {property_text}`
- Each correctness property must be implemented by a single property-based test

### Unit Testing Focus

Unit tests should focus on:
- Specific examples that demonstrate correct behavior (e.g., known text → expected embedding dimension)
- Integration points between components (e.g., cache integration with embedding generator)
- Edge cases (e.g., exactly at token limit, special characters)
- Error conditions (e.g., network failures, invalid JSON)

Avoid writing too many unit tests for scenarios that property tests already cover comprehensively.

### Test Coverage Goals

- **Code Coverage**: Minimum 85% line coverage
- **Property Coverage**: All 25 correctness properties implemented as property tests
- **Error Path Coverage**: All error types and retry scenarios tested
- **Integration Coverage**: End-to-end flows from text input to similarity score

### Testing Layers

1. **Unit Tests**:
   - Individual component behavior (EmbeddingGenerator, SimilarityComputer, Cache)
   - Mocked dependencies
   - Fast execution (<100ms per test)

2. **Property Tests**:
   - Universal properties across random inputs
   - Real component integration (no mocks for core logic)
   - 100+ iterations per property
   - Moderate execution time (<5s per property)

3. **Integration Tests**:
   - End-to-end flows with real or stubbed embedding models
   - Cache behavior across multiple operations
   - Batch processing with various sizes
   - Error handling and retry logic

4. **Performance Tests**:
   - Cache hit rate under realistic workloads
   - Batch processing efficiency
   - Memory usage with large caches
   - Concurrent request handling

### Example Property Test Structure

```python
# Example using Hypothesis (Python)
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=1, max_size=1000).filter(lambda t: t.strip())
)
def test_property_1_valid_text_produces_embeddings(text):
    """
    Feature: semantic-similarity-scoring-engine
    Property 1: Valid text produces embeddings
    
    For any valid non-empty text input, the embedding generation 
    should produce a numerical vector with consistent dimension.
    """
    engine = ScoringEngine(EngineConfig.default())
    embedding = engine.embed_text(text)
    
    assert embedding is not None
    assert len(embedding.vector) == engine.config.expected_dimension
    assert all(isinstance(v, float) for v in embedding.vector)
    assert embedding.model_name == engine.config.model_name
```

### Mock Strategy

- **Mock external APIs**: Embedding model APIs should be mocked in unit tests
- **Real implementations for properties**: Property tests should use real logic (not mocked) to verify correctness
- **Stub for integration tests**: Use lightweight embedding models or stubs for integration tests
- **No mocks for pure functions**: Functions like cosine similarity should never be mocked

### Continuous Integration

- All tests run on every commit
- Property tests run with reduced iterations (50) in CI, full iterations (100+) nightly
- Performance tests run nightly or on-demand
- Test results tracked over time to detect regressions
