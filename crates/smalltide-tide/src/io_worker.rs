//! Dedicated background I/O worker thread for Stage 2 (`NVMe Disk -> Tier 1 RAM`).
//!
//! Keeps Win32 `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` reads completely off the
//! GPU decode loop so `0.96 ms` NVMe reads overlap with token generation.

use crate::cache::ExpertClass;
use smalltide_core::{ExpertId, Result, SlotIndex};
use smalltide_io::stide_pack::StidePackReader;
use smalltide_io::win_direct_io::AlignedBuffer;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

pub struct CompletedDiskRead {
    pub expert: ExpertId,
    pub class: ExpertClass,
    pub ram_slot: SlotIndex,
    pub payload: Result<Vec<u8>>,
}

enum IoCommand {
    /// Non-blocking Stage 2 background read into a Tier 1 RAM slot.
    AsyncRead {
        expert: ExpertId,
        class: ExpertClass,
        ram_slot: SlotIndex,
        slab_len: usize,
    },
    /// Synchronous read (cold start or blocking sync rescue): served before queued `AsyncRead`s.
    SyncRead {
        expert: ExpertId,
        slab_len: usize,
        reply: Sender<Result<Vec<u8>>>,
    },
    Shutdown,
}

/// Persistent single-threaded NVMe direct-I/O worker.
pub struct Stage2IoWorker {
    tx: Sender<IoCommand>,
    rx_done: Receiver<CompletedDiskRead>,
    handle: Option<JoinHandle<()>>,
    in_flight: usize,
}

impl Stage2IoWorker {
    pub fn spawn(pack: StidePackReader, max_slab_bytes: usize) -> Result<Self> {
        let (tx, rx_cmd) = mpsc::channel::<IoCommand>();
        let (tx_done, rx_done) = mpsc::channel::<CompletedDiskRead>();

        let mut scratch = AlignedBuffer::alloc(max_slab_bytes)?;
        let handle = thread::Builder::new()
            .name("smalltide-stage2-nvme".into())
            .spawn(move || {
                // A synchronous read has the GPU waiting for it (a rescue of an expert that is
                // on the disk only): it goes ahead of every queued background read, so it waits
                // for at most the read in progress. In arrival order it waited behind all of
                // them (seen: 20 ms a read with a hundred background reads queued).
                let mut queue: std::collections::VecDeque<IoCommand> = std::collections::VecDeque::new();
                loop {
                    if queue.is_empty() {
                        match rx_cmd.recv() {
                            Ok(cmd) => queue.push_back(cmd),
                            Err(_) => break,
                        }
                    }
                    while let Ok(cmd) = rx_cmd.try_recv() {
                        queue.push_back(cmd);
                    }
                    let first = queue
                        .iter()
                        .position(|c| matches!(c, IoCommand::SyncRead { .. } | IoCommand::Shutdown))
                        .unwrap_or(0);
                    let Some(cmd) = queue.remove(first) else { break };
                    match cmd {
                        IoCommand::AsyncRead {
                            expert,
                            class,
                            ram_slot,
                            slab_len,
                        } => {
                            let res = pack
                                .read_expert_direct(expert, &mut scratch)
                                .map(|_| scratch.as_slice()[..slab_len].to_vec());
                            let _ = tx_done.send(CompletedDiskRead {
                                expert,
                                class,
                                ram_slot,
                                payload: res,
                            });
                        }
                        IoCommand::SyncRead {
                            expert,
                            slab_len,
                            reply,
                        } => {
                            let res = pack
                                .read_expert_direct(expert, &mut scratch)
                                .map(|_| scratch.as_slice()[..slab_len].to_vec());
                            let _ = reply.send(res);
                        }
                        IoCommand::Shutdown => break,
                    }
                }
            })?;

        Ok(Self {
            tx,
            rx_done,
            handle: Some(handle),
            in_flight: 0,
        })
    }

    /// Enqueues an asynchronous Stage 2 NVMe read without blocking the decode loop.
    pub fn submit_async_read(
        &mut self,
        expert: ExpertId,
        class: ExpertClass,
        ram_slot: SlotIndex,
        slab_len: usize,
    ) {
        if self
            .tx
            .send(IoCommand::AsyncRead {
                expert,
                class,
                ram_slot,
                slab_len,
            })
            .is_ok()
        {
            self.in_flight += 1;
        }
    }

    /// Non-blocking drain of all completed background Stage 2 NVMe reads.
    pub fn poll_completed(&mut self) -> Vec<CompletedDiskRead> {
        let mut out = Vec::new();
        while let Ok(item) = self.rx_done.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            out.push(item);
        }
        out
    }

    /// Blocking read for cold start or sync rescue when an expert is missing from both VRAM and RAM.
    pub fn read_sync(&self, expert: ExpertId, slab_len: usize) -> Result<Vec<u8>> {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(IoCommand::SyncRead {
                expert,
                slab_len,
                reply: rtx,
            })
            .map_err(|e| {
                smalltide_core::SmallTideError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    format!("Stage2IoWorker disconnected: {e}"),
                ))
            })?;
        rrx.recv().map_err(|e| {
            smalltide_core::SmallTideError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                format!("Stage2IoWorker reply failed: {e}"),
            ))
        })?
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }
}

impl Drop for Stage2IoWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(IoCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
