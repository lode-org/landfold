//! Provenance required for reproducible Landfold result artifacts.

/// Version of the provenance object embedded in Landfold result artifacts.
pub const PROVENANCE_SCHEMA: &str = "landfold.provenance.v1";

/// Compatibility and input identity for a Landfold analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// Versioned provenance schema.
    pub schema: &'static str,
    /// Stable identifier for the anneal run or source trajectory.
    pub run_id: String,
    /// SHA-256 digest of the exact source artifact, prefixed with `sha256:`.
    pub input_digest: String,
    /// Producer identifier, normally `rgpot` or a named trajectory source.
    pub engine_id: String,
    /// Exact `eindir` source revision for objective-engine producers.
    pub eindir_revision: Option<String>,
    /// Producer protocol family.
    pub protocol_family: String,
    /// Wire-incompatible producer protocol revision.
    pub protocol_major: u16,
    /// Additive producer protocol revision.
    pub protocol_minor: u16,
    /// Embedded eindir objective layout revision.
    pub abi_layout_revision: u32,
    /// DLPack major revision used by the source bridge.
    pub dlpack_major: u16,
    /// DLPack minor revision used by the source bridge.
    pub dlpack_minor: u16,
}

impl Provenance {
    /// Construct and validate a source provenance record.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        input_digest: impl Into<String>,
        engine_id: impl Into<String>,
        protocol_family: impl Into<String>,
        protocol_major: u16,
        protocol_minor: u16,
        abi_layout_revision: u32,
        dlpack_major: u16,
        dlpack_minor: u16,
    ) -> Result<Self, String> {
        Self::from_fields(
            run_id,
            input_digest,
            engine_id,
            None,
            protocol_family,
            protocol_major,
            protocol_minor,
            abi_layout_revision,
            dlpack_major,
            dlpack_minor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_fields(
        run_id: impl Into<String>,
        input_digest: impl Into<String>,
        engine_id: impl Into<String>,
        eindir_revision: Option<String>,
        protocol_family: impl Into<String>,
        protocol_major: u16,
        protocol_minor: u16,
        abi_layout_revision: u32,
        dlpack_major: u16,
        dlpack_minor: u16,
    ) -> Result<Self, String> {
        let provenance = Self {
            schema: PROVENANCE_SCHEMA,
            run_id: run_id.into(),
            input_digest: input_digest.into(),
            engine_id: engine_id.into(),
            eindir_revision,
            protocol_family: protocol_family.into(),
            protocol_major,
            protocol_minor,
            abi_layout_revision,
            dlpack_major,
            dlpack_minor,
        };
        provenance.validate()?;
        Ok(provenance)
    }

    /// Construct a validated record with the exact objective-engine source revision.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_eindir_revision(
        run_id: impl Into<String>,
        input_digest: impl Into<String>,
        engine_id: impl Into<String>,
        protocol_family: impl Into<String>,
        protocol_major: u16,
        protocol_minor: u16,
        abi_layout_revision: u32,
        dlpack_major: u16,
        dlpack_minor: u16,
        eindir_revision: impl Into<String>,
    ) -> Result<Self, String> {
        Self::from_fields(
            run_id,
            input_digest,
            engine_id,
            Some(eindir_revision.into()),
            protocol_family,
            protocol_major,
            protocol_minor,
            abi_layout_revision,
            dlpack_major,
            dlpack_minor,
        )
    }

    /// Validate fields that make the artifact joinable to a source run.
    pub fn validate(&self) -> Result<(), String> {
        if self.run_id.trim().is_empty() {
            return Err("run ID must not be empty".into());
        }
        let digest = self
            .input_digest
            .strip_prefix("sha256:")
            .ok_or_else(|| "input digest must use the sha256: prefix".to_owned())?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("input digest must contain 64 hexadecimal SHA-256 digits".into());
        }
        if self.engine_id.trim().is_empty() || self.protocol_family.trim().is_empty() {
            return Err("engine ID and protocol family must not be empty".into());
        }
        if let Some(revision) = self.eindir_revision.as_deref() {
            if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("eindir revision must be a 40-digit hexadecimal commit".into());
            }
        } else if self.engine_id == "rgpot" {
            return Err("rgpot provenance must include the eindir revision".into());
        }
        if self.protocol_major == 0 || self.abi_layout_revision == 0 || self.dlpack_major == 0 {
            return Err("protocol, ABI, and DLPack major revisions must be nonzero".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_complete_anneal_engine_provenance_record() {
        let provenance = Provenance::new_with_eindir_revision(
            "run-42",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "rgpot",
            "rgpot.potentials",
            1,
            0,
            1,
            1,
            0,
            "f3c42130bb389ba6cd6e4cfdc8b2e182f4a764e9",
        )
        .expect("complete provenance should be accepted");
        assert_eq!(provenance.schema, PROVENANCE_SCHEMA);
        assert_eq!(provenance.run_id, "run-42");
        assert_eq!(provenance.engine_id, "rgpot");
    }

    #[test]
    fn rejects_rgpot_provenance_without_an_eindir_revision() {
        let error = Provenance::new(
            "run-42",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "rgpot",
            "rgpot.potentials",
            1,
            0,
            1,
            1,
            0,
        )
        .expect_err("rgpot provenance must identify its eindir source");
        assert!(error.contains("eindir revision"));
    }

    #[test]
    fn rejects_malformed_eindir_revision() {
        let error = Provenance::new_with_eindir_revision(
            "run-42",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "rgpot",
            "rgpot.potentials",
            1,
            0,
            1,
            1,
            0,
            "not-a-commit",
        )
        .expect_err("provenance must carry a complete commit revision");
        assert!(error.contains("40-digit hexadecimal"));
    }

    #[test]
    fn rejects_a_provenance_record_without_a_digest() {
        let error = Provenance::new("run-42", "", "rgpot", "rgpot.potentials", 1, 0, 1, 1, 0)
            .expect_err("artifact provenance must identify its input");
        assert!(error.contains("input digest"));
    }
}
