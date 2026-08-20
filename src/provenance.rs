//! Provenance required for reproducible Landfold result artifacts.

use serde_json::{Value, json};

/// Version of the provenance object embedded in Landfold result artifacts.
pub const PROVENANCE_SCHEMA: &str = "landfold.provenance.v1";
pub const EON_COMPATIBILITY_SCHEMA: &str = "eon.compatibility.v1";

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
    /// Embedded eindir objective ABI major revision.
    pub abi_major: u16,
    /// Embedded eindir objective ABI minor revision.
    pub abi_minor: u16,
    /// DLPack major revision used by the source bridge.
    pub dlpack_major: u16,
    /// DLPack minor revision used by the source bridge.
    pub dlpack_minor: u16,
}

/// Compatibility stamp carried by eOn's `EngineCompatibility` Cap'n Proto
/// record and its JSON representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineCompatibility {
    pub schema: String,
    pub engine_id: String,
    pub protocol_family: String,
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub abi_major: u16,
    pub abi_minor: u16,
    pub layout_revision: u32,
    pub build_identity: String,
    pub readcon_spec_version: Option<u16>,
    pub readcon_min_version: Option<String>,
    pub eon_schema_min_version: Option<String>,
    pub rgpycrumbs_min_version: Option<String>,
    pub chemparseplot_min_version: Option<String>,
}

impl EngineCompatibility {
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "eOn engine compatibility must be an object".to_owned())?;
        let text = |key: &str| -> Result<String, String> {
            object
                .get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("eOn engine compatibility requires string field {key}"))
        };
        let integer = |key: &str| -> Result<u64, String> {
            object
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("eOn engine compatibility requires integer field {key}"))
        };
        let optional_integer = |key: &str| -> Result<Option<u16>, String> {
            object
                .get(key)
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| {
                            format!("eOn engine compatibility field {key} must be an integer")
                        })
                        .and_then(|value| {
                            u16::try_from(value).map_err(|_| {
                                format!("eOn engine compatibility field {key} exceeds UInt16")
                            })
                        })
                })
                .transpose()
        };
        let optional_text = |key: &str| -> Result<Option<String>, String> {
            object
                .get(key)
                .map(|value| {
                    value
                        .as_str()
                        .filter(|value| !value.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or_else(|| {
                            format!(
                                "eOn engine compatibility field {key} must be a non-empty string"
                            )
                        })
                })
                .transpose()
        };
        let schema = text("schema")?;
        if schema != EON_COMPATIBILITY_SCHEMA {
            return Err(format!("unsupported eOn compatibility schema {schema}"));
        }
        let protocol_major = integer("protocolMajor")?;
        let protocol_minor = integer("protocolMinor")?;
        let abi_major = integer("abiMajor")?;
        let abi_minor = integer("abiMinor")?;
        let layout_revision = integer("layoutRevision")?;
        Ok(Self {
            schema,
            engine_id: text("engineId")?,
            protocol_family: text("protocolFamily")?,
            protocol_major: u16::try_from(protocol_major)
                .map_err(|_| "protocolMajor exceeds UInt16".to_owned())?,
            protocol_minor: u16::try_from(protocol_minor)
                .map_err(|_| "protocolMinor exceeds UInt16".to_owned())?,
            abi_major: u16::try_from(abi_major)
                .map_err(|_| "abiMajor exceeds UInt16".to_owned())?,
            abi_minor: u16::try_from(abi_minor)
                .map_err(|_| "abiMinor exceeds UInt16".to_owned())?,
            layout_revision: u32::try_from(layout_revision)
                .map_err(|_| "layoutRevision exceeds UInt32".to_owned())?,
            build_identity: text("buildIdentity")?,
            readcon_spec_version: optional_integer("readconSpecVersion")?,
            readcon_min_version: optional_text("readconMinVersion")?,
            eon_schema_min_version: optional_text("eonSchemaMinVersion")?,
            rgpycrumbs_min_version: optional_text("rgpycrumbsMinVersion")?,
            chemparseplot_min_version: optional_text("chemparseplotMinVersion")?,
        })
    }

    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "schema": self.schema,
            "engineId": self.engine_id,
            "protocolFamily": self.protocol_family,
            "protocolMajor": self.protocol_major,
            "protocolMinor": self.protocol_minor,
            "abiMajor": self.abi_major,
            "abiMinor": self.abi_minor,
            "layoutRevision": self.layout_revision,
            "buildIdentity": self.build_identity,
        });
        if let Some(version) = self.readcon_spec_version {
            value["readconSpecVersion"] = json!(version);
        }
        if let Some(version) = &self.readcon_min_version {
            value["readconMinVersion"] = json!(version);
        }
        if let Some(version) = &self.eon_schema_min_version {
            value["eonSchemaMinVersion"] = json!(version);
        }
        if let Some(version) = &self.rgpycrumbs_min_version {
            value["rgpycrumbsMinVersion"] = json!(version);
        }
        if let Some(version) = &self.chemparseplot_min_version {
            value["chemparseplotMinVersion"] = json!(version);
        }
        value
    }
}

impl Provenance {
    /// Convert an eOn engine stamp into Landfold's artifact provenance.
    pub fn from_eon_compatibility(
        run_id: impl Into<String>,
        input_digest: impl Into<String>,
        compatibility: &EngineCompatibility,
    ) -> Result<Self, String> {
        Self::from_fields(
            run_id,
            input_digest,
            &compatibility.engine_id,
            None,
            &compatibility.protocol_family,
            compatibility.protocol_major,
            compatibility.protocol_minor,
            compatibility.layout_revision,
            compatibility.abi_major,
            compatibility.abi_minor,
            1,
            0,
        )
    }

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
            1,
            0,
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
        abi_major: u16,
        abi_minor: u16,
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
            abi_major,
            abi_minor,
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
            1,
            0,
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
        if self.protocol_major == 0
            || self.abi_major == 0
            || self.abi_layout_revision == 0
            || self.dlpack_major == 0
        {
            return Err("protocol, ABI, and DLPack major revisions must be nonzero".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eon_stamp() -> Value {
        serde_json::json!({
            "schema": "eon.compatibility.v1",
            "engineId": "eon",
            "protocolFamily": "eon.objective",
            "protocolMajor": 1,
            "protocolMinor": 0,
            "abiMajor": 1,
            "abiMinor": 1,
            "layoutRevision": 3,
            "buildIdentity": "eon-2.11.1+abc123"
        })
    }

    #[test]
    fn eon_engine_stamp_round_trips_stack_floor_fields() {
        let mut stamp = eon_stamp();
        stamp["readconSpecVersion"] = json!(3);
        stamp["readconMinVersion"] = json!("0.14.7");
        stamp["eonSchemaMinVersion"] = json!("0.2.0");
        stamp["rgpycrumbsMinVersion"] = json!("1.10.4");
        stamp["chemparseplotMinVersion"] = json!("1.9.17");
        let compatibility = EngineCompatibility::from_json(&stamp).unwrap();
        assert_eq!(compatibility.readcon_spec_version, Some(3));
        assert_eq!(compatibility.to_json(), stamp);
    }

    #[test]
    fn eon_engine_stamp_round_trips_into_landfold_provenance() {
        let compatibility = EngineCompatibility::from_json(&eon_stamp()).unwrap();
        let provenance = Provenance::from_eon_compatibility(
            "run-42",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &compatibility,
        )
        .unwrap();
        assert_eq!(provenance.engine_id, "eon");
        assert_eq!(provenance.abi_major, 1);
        assert_eq!(provenance.abi_minor, 1);
        assert_eq!(provenance.abi_layout_revision, 3);
        assert_eq!(provenance.dlpack_major, 1);
        assert_eq!(provenance.dlpack_minor, 0);
        assert_eq!(compatibility.to_json(), eon_stamp());
    }

    #[test]
    fn eon_engine_stamp_rejects_wrong_schema() {
        let mut stamp = eon_stamp();
        stamp["schema"] = Value::String("eon.compatibility.v0".into());
        let error = EngineCompatibility::from_json(&stamp).unwrap_err();
        assert!(error.contains("unsupported eOn compatibility schema"));
    }

    #[test]
    fn accepts_a_complete_anneal_engine_provenance_record() {
        let provenance = Provenance::new_with_eindir_revision(
            "run-42",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "rgpot",
            "rgpot.potentials",
            1,
            0,
            3,
            1,
            0,
            "091c6f7d6ea70821c3374152481fb9500e799af2",
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
