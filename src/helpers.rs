//! Helper utilities for agent operation
//!
//! This module provides common utility functions used throughout the agent system,
//! matching the structure from go-agent's helpers.go.

/// Sanitizes user input to prevent prompt injection
pub fn sanitize_input(s: &str) -> String {
    let mut result = s.trim().to_string();
    result = result.replace("\nUser:", "\nUser (quoted):");
    result = result.replace("\nSystem:", "\nSystem (quoted):");
    result = result.replace("\nConversation memory", "\nConversation memory (quoted)");
    result
}

/// Escapes content for safe inclusion in prompts
pub fn escape_prompt_content(s: &str) -> String {
    let mut result = s.replace('`', "'");
    result = result.replace("\nUser:", "\nUser (quoted):");
    result = result.replace("\nSystem:", "\nSystem (quoted):");
    result
}

/// Extracts JSON from a string that may contain additional text
pub fn extract_json(s: &str) -> Option<String> {
    let trimmed = s.trim();
    
    // Try to find JSON object or array
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                return Some(trimmed[start..=end].to_string());
            }
        }
    }
    
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if end > start {
                return Some(trimmed[start..=end].to_string());
            }
        }
    }
    
    // If the whole string looks like JSON, return it
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        return Some(trimmed.to_string());
    }
    
    None
}

/// Validates that a code snippet is safe to execute
pub fn is_valid_snippet(code: &str) -> bool {
    let code = code.trim();
    
    if code.is_empty() {
        return false;
    }
    
    // Reject obviously dangerous patterns
    let dangerous = [
        "rm -rf",
        "format c:",
        "del /f",
        "DROP DATABASE",
        "DROP TABLE",
    ];
    
    let lower = code.to_lowercase();
    for pattern in dangerous {
        if lower.contains(&pattern.to_lowercase()) {
            return false;
        }
    }
    
    true
}

/// Splits a command string into name and arguments
pub fn split_command(s: &str) -> (&str, &str) {
    let trimmed = s.trim();
    if let Some(pos) = trimmed.find(char::is_whitespace) {
        let (name, rest) = trimmed.split_at(pos);
        (name.trim(), rest.trim())
    } else {
        (trimmed, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_input() {
        let input = "Hello\nUser: inject\nSystem: bad";
        let sanitized = sanitize_input(input);
        assert!(!sanitized.contains("\nUser:"));
        assert!(!sanitized.contains("\nSystem:"));
        assert!(sanitized.contains("User (quoted)"));
    }

    #[test]
    fn test_escape_prompt_content() {
        let content = "`code`\nUser: test";
        let escaped = escape_prompt_content(content);
        assert!(!escaped.contains('`'));
        assert!(escaped.contains("User (quoted)"));
    }

    #[test]
    fn test_extract_json() {
        assert_eq!(
            extract_json("Some text {\"key\": \"value\"} more text"),
            Some("{\"key\": \"value\"}".to_string())
        );
        assert_eq!(
            extract_json("[1, 2, 3]"),
            Some("[1, 2, 3]".to_string())
        );
        assert_eq!(extract_json("no json here"), None);
    }

    #[test]
    fn test_is_valid_snippet() {
        assert!(is_valid_snippet("let x = 5;"));
        assert!(!is_valid_snippet("rm -rf /"));
        assert!(!is_valid_snippet("DROP DATABASE users;"));
        assert!(!is_valid_snippet(""));
    }

    #[test]
    fn test_split_command() {
        assert_eq!(split_command("echo hello world"), ("echo", "hello world"));
        assert_eq!(split_command("tool"), ("tool", ""));
        assert_eq!(split_command("  cmd  args  "), ("cmd", "args"));
    }
}
