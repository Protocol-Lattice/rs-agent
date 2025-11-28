//! Query classification utilities
//!
//! This module provides query type detection to optimize context retrieval,
//! matching the structure from go-agent's query.go.

/// Types of queries that determine retrieval strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Mathematical or computational queries that don't need context
    Math,
    /// Short factoid questions needing minimal context
    ShortFactoid,
    /// Complex queries requiring full context retrieval
    Complex,
    /// Unknown query type, skip retrieval
    Unknown,
}

/// Classifies a query to determine optimal retrieval strategy
pub fn classify_query(query: &str) -> QueryType {
    let lower = query.trim().to_lowercase();
    
    // Math queries - numerical operations, calculations
    if is_math_query(&lower) {
        return QueryType::Math;
    }
    
    // Short factoid queries - simple definitions, factual questions
    if is_short_factoid(&lower) {
        return QueryType::ShortFactoid;
    }
    
    // Complex queries - explanations, multi-step reasoning
    if is_complex_query(&lower) {
        return QueryType::Complex;
    }
    
    QueryType::Unknown
}

fn is_math_query(query: &str) -> bool {
    // Mathematical operators
    let has_math_ops = query.contains('+')
        || query.contains('-')
        || query.contains('*')
        || query.contains('/')
        || query.contains('^')
        || query.contains('=');
    
    // Mathematical keywords
    let math_keywords = [
        "calculate",
        "compute",
        "solve",
        "equation",
        "sum",
        "multiply",
        "divide",
        "subtract",
        "add",
        "integral",
        "derivative",
    ];
    
    has_math_ops || math_keywords.iter().any(|&kw| query.contains(kw))
}

fn is_short_factoid(query: &str) -> bool {
    // Short questions typically start with question words
    let question_starts = [
        "what is",
        "who is",
        "when was",
        "where is",
        "which",
        "define",
    ];
    
    // Must be relatively short
    let word_count = query.split_whitespace().count();
    
    question_starts.iter().any(|&start| query.starts_with(start)) 
        && word_count < 15
}

fn is_complex_query(query: &str) -> bool {
    // Complexity indicators
    let complex_keywords = [
        "explain",
        "describe",
        "analyze",
        "compare",
        "discuss",
        "evaluate",
        "how does",
        "why does",
        "tell me about",
        "walk me through",
    ];
    
    // Longer queries are typically more complex
    let word_count = query.split_whitespace().count();
    
    complex_keywords.iter().any(|&kw| query.contains(kw))
        || word_count > 20
        || query.contains('?') && word_count > 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_math_queries() {
        assert_eq!(classify_query("What is 5 + 3?"), QueryType::Math);
        assert_eq!(classify_query("Calculate the sum of 10 and 20"), QueryType::Math);
        assert_eq!(classify_query("Solve x^2 = 4"), QueryType::Math);
    }

    #[test]
    fn test_classify_short_factoid() {
        assert_eq!(classify_query("What is Rust?"), QueryType::ShortFactoid);
        assert_eq!(classify_query("Who is the president?"), QueryType::ShortFactoid);
        assert_eq!(classify_query("When was Python created?"), QueryType::ShortFactoid);
    }

    #[test]
    fn test_classify_complex_queries() {
        assert_eq!(
            classify_query("Explain how async/await works in Rust"),
            QueryType::Complex
        );
        assert_eq!(
            classify_query("Tell me about the history of programming languages and their evolution over time"),
            QueryType::Complex
        );
        assert_eq!(
            classify_query("Why does the borrow checker prevent certain patterns?"),
            QueryType::Complex
        );
    }

    #[test]
    fn test_classify_unknown() {
        assert_eq!(classify_query("Hello"), QueryType::Unknown);
        assert_eq!(classify_query(""), QueryType::Unknown);
    }

    #[test]
    fn test_edge_cases() {
        // "What is" but too long for short factoid
        let long_what = "What is the meaning of life and how do we determine our purpose in this vast universe?";
        assert_eq!(classify_query(long_what), QueryType::Complex);
        
        // Math with explanation request
        assert_eq!(
            classify_query("Explain how to solve quadratic equations"),
            QueryType::Complex
        );
    }
}
