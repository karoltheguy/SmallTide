use serde::Deserialize;
use smalltide_core::{Result, SmallTideError};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorHeaderEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    pub header_len: u64,
    pub tensors: BTreeMap<String, TensorHeaderEntry>,
}

/// Reads only the JSON header of a `.safetensors` file without loading tensor data.
pub fn inspect_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > 256 * 1024 * 1024 {
        return Err(SmallTideError::InvalidModel(format!(
            "safetensors header too large: {header_len} bytes"
        )));
    }
    let mut json_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut json_buf)?;

    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&json_buf)
        .map_err(|e| SmallTideError::InvalidModel(format!("invalid safetensors JSON: {e}")))?;

    let mut tensors = BTreeMap::new();
    for (name, val) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: TensorHeaderEntry = serde_json::from_value(val).map_err(|e| {
            SmallTideError::InvalidModel(format!("invalid tensor entry '{name}': {e}"))
        })?;
        tensors.insert(name, entry);
    }

    Ok(SafetensorsHeader {
        header_len,
        tensors,
    })
}
