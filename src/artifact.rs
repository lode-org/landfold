//! Versioned result-schema identifiers shared by language bindings and tools.

/// Schema identifier for embedding results.
pub const EMBEDDING_SCHEMA: &str = "landfold.embedding.v1";
/// Schema identifier for free-energy surface results.
pub const FES_SCHEMA: &str = "landfold.fes.v1";
/// Schema identifier for out-of-sample projection results.
pub const PROJECTION_SCHEMA: &str = "landfold.projection.v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_are_versioned() {
        for schema in [EMBEDDING_SCHEMA, FES_SCHEMA, PROJECTION_SCHEMA] {
            assert!(schema.ends_with(".v1"));
        }
    }
}
