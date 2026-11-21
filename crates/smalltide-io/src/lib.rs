//! Stage 2 storage I/O: Win32 `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` / IoRing
//! async expert reader, Safetensors & GGUF parsers, and optional 4 KB-aligned expert packer.

pub mod dense_loader;
pub mod quant;
pub mod safetensors;
pub mod stide_pack;
pub mod tokenizer;
pub mod win_direct_io;
