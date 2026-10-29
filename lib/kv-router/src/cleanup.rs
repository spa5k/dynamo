// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared stale-child cleanup machinery for rooted tree structures.
//!
//! Provides a throttled, one-in-flight sweep that unlinks empty child nodes
//! from their parent. It is used by the sequence-side
//! [`PromptMembershipTrie`](super::sequences::prompt_membership_trie::PromptMembershipTrie),
//! which embeds a [`CleanupState`] and implements [`CleanableNode`] for its
//! node type.
//!
//! # Sweep semantics
//!
//! [`sweep_stale_children`] is a reverse-BFS prune:
//! - BFS from the root under read locks, collecting `(parent_weak, key, child_weak)` edges.
//! - Iterate edges deepest-first so children are swept before parents.
//! - For each parent: upgrade weaks, keep the children that look stale, then
//!   take the parent write lock once. For each kept child, verify the child
//!   pointer still matches, `try_write` the child, and unlink only when
//!   the child has no workers, no children, and `Arc::strong_count == 2`
//!   (parent map ref + our local upgrade). The strong-count gate is what
//!   prevents reclaiming a node that a concurrent `find_matches` is currently
//!   traversing — such edges are skipped and retried on the next sweep.
//!
//! # Sweep schedule
//!
//! A sweep is due when [`CLEANUP_INTERVAL_MS`] has passed, or earlier when the
//! nodes that lost their last holder since the last sweep reach a budget. The
//! budget is [`DEATHS_PER_KEPT_NODE`] times the nodes the last sweep kept, and
//! at least [`MIN_DEATHS_BEFORE_SWEEP`]. Dead nodes stay in place until then, so
//! a later request on the same prefix can revive them. The budget keeps the
//! dead nodes, and thus each sweep, proportional to the live tree.

use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

pub const CLEANUP_INTERVAL_MS: u64 = 5 * 60 * 1000;
/// Deaths since the last sweep that make a sweep due, per node the last sweep kept.
pub const DEATHS_PER_KEPT_NODE: usize = 2;
/// Deaths since the last sweep that always make a sweep due.
pub const MIN_DEATHS_BEFORE_SWEEP: usize = 4096;

/// Node type that participates in the reverse-BFS cleanup sweep.
pub trait CleanableNode: Sized + Send + Sync + 'static {
    /// Key type used in this node's children map (e.g. `LocalBlockHash`,
    /// `SequenceHash`).
    type ChildKey: Copy + Eq + Hash + Send + Sync + 'static;

    /// True if this node still carries worker state that pins it in the tree.
    fn has_any_workers(&self) -> bool;

    /// Read-only view of this node's children keyed by the first edge element.
    fn children(&self) -> &FxHashMap<Self::ChildKey, Arc<RwLock<Self>>>;

    /// Unlink a child edge.
    fn remove_child(&mut self, key: &Self::ChildKey);
}

pub struct CleanupState {
    clock_origin: Instant,
    last_cleanup_elapsed_ms: AtomicU64,
    scheduled: AtomicBool,
    /// Nodes that lost their last holder since the last sweep began.
    deaths_since_sweep: AtomicUsize,
    /// Nodes that the last completed sweep kept.
    kept_by_last_sweep: AtomicUsize,
    /// Set when `deaths_since_sweep` reaches the budget, so that
    /// [`Self::try_schedule`] reads one flag instead of the contended counter.
    over_budget: AtomicBool,
}

impl CleanupState {
    pub fn new() -> Self {
        Self {
            clock_origin: Instant::now(),
            last_cleanup_elapsed_ms: AtomicU64::new(0),
            scheduled: AtomicBool::new(false),
            deaths_since_sweep: AtomicUsize::new(0),
            kept_by_last_sweep: AtomicUsize::new(0),
            over_budget: AtomicBool::new(false),
        }
    }

    /// Count a node that lost its last holder toward the next sweep.
    pub fn note_death(&self) {
        let deaths = self.deaths_since_sweep.fetch_add(1, Ordering::Relaxed) + 1;
        let budget = (self.kept_by_last_sweep.load(Ordering::Relaxed) * DEATHS_PER_KEPT_NODE)
            .max(MIN_DEATHS_BEFORE_SWEEP);
        if deaths >= budget && !self.over_budget.load(Ordering::Relaxed) {
            self.over_budget.store(true, Ordering::Relaxed);
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.clock_origin.elapsed().as_millis() as u64
    }

    pub fn try_schedule(&self) -> bool {
        if !self.over_budget.load(Ordering::Relaxed) {
            let now_ms = self.elapsed_ms();
            let last_ms = self.last_cleanup_elapsed_ms.load(Ordering::Relaxed);
            if now_ms.saturating_sub(last_ms) < CLEANUP_INTERVAL_MS {
                return false;
            }
        }

        let scheduled = self
            .scheduled
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok();
        if scheduled {
            // Deaths during the sweep count toward the next one.
            self.over_budget.store(false, Ordering::Relaxed);
            self.deaths_since_sweep.store(0, Ordering::Relaxed);
        }
        scheduled
    }
}

impl Default for CleanupState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CleanupGuard<'a> {
    state: &'a CleanupState,
    completed_elapsed_ms: Option<u64>,
    kept: usize,
}

impl<'a> CleanupGuard<'a> {
    pub fn new(state: &'a CleanupState) -> Self {
        Self {
            state,
            completed_elapsed_ms: None,
            kept: 0,
        }
    }

    /// Record a finished sweep that kept `kept` nodes.
    pub fn mark_completed(&mut self, kept: usize) {
        self.completed_elapsed_ms = Some(self.state.elapsed_ms());
        self.kept = kept;
    }
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        if let Some(elapsed_ms) = self.completed_elapsed_ms {
            self.state
                .last_cleanup_elapsed_ms
                .store(elapsed_ms, Ordering::Relaxed);
            self.state
                .kept_by_last_sweep
                .store(self.kept, Ordering::Relaxed);
        }
        self.state.scheduled.store(false, Ordering::Release);
    }
}

struct CleanupEdge<N: CleanableNode> {
    parent: Weak<RwLock<N>>,
    key: N::ChildKey,
    child: Weak<RwLock<N>>,
}

/// Reverse-BFS sweep that unlinks empty, unreferenced leaf nodes from the tree.
/// Returns how many non-root nodes it kept.
pub fn sweep_stale_children<N: CleanableNode>(root: &Arc<RwLock<N>>) -> usize {
    let mut queue: VecDeque<Arc<RwLock<N>>> = VecDeque::from([root.clone()]);
    let mut edges: Vec<CleanupEdge<N>> = Vec::new();

    while let Some(parent) = queue.pop_front() {
        let guard = parent.read();
        for (&key, child) in guard.children() {
            queue.push_back(child.clone());
            edges.push(CleanupEdge {
                parent: Arc::downgrade(&parent),
                key,
                child: Arc::downgrade(child),
            });
        }
    }

    // The walk lists one parent's children together, so each parent's edges
    // are contiguous, and later parents are deeper. Unlink each parent's
    // stale children under one write lock, deepest parents first.
    let mut kept = edges.len();
    let mut end = edges.len();
    while end > 0 {
        let parent = edges[end - 1].parent.as_ptr();
        let mut start = end - 1;
        while start > 0 && std::ptr::eq(edges[start - 1].parent.as_ptr(), parent) {
            start -= 1;
        }
        kept -= unlink_stale_children(&edges[start..end]);
        end = start;
    }
    kept
}

/// Unlink the stale children of one parent and return how many of `edges`
/// are gone: unlinked here, or already dropped.
fn unlink_stale_children<N: CleanableNode>(edges: &[CleanupEdge<N>]) -> usize {
    let Some(parent) = edges[0].parent.upgrade() else {
        return edges.len();
    };
    let mut gone = 0;
    // Lock the parent only for children that look stale now.
    let candidates: Vec<_> = edges
        .iter()
        .filter_map(|edge| {
            let Some(child) = edge.child.upgrade() else {
                gone += 1;
                return None;
            };
            let stale = child
                .try_read()
                .is_some_and(|guard| !guard.has_any_workers() && guard.children().is_empty());
            stale.then_some((edge.key, child))
        })
        .collect();
    if candidates.is_empty() {
        return gone;
    }

    let mut unlinked = Vec::with_capacity(candidates.len());
    let mut parent_guard = parent.write();
    for (key, child) in candidates {
        let still_attached = parent_guard
            .children()
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, &child));
        if !still_attached {
            continue;
        }

        let Some(child_guard) = child.try_write() else {
            continue;
        };
        if child_guard.has_any_workers() || !child_guard.children().is_empty() {
            continue;
        }
        if Arc::strong_count(&child) != 2 {
            continue;
        }

        parent_guard.remove_child(&key);
        drop(child_guard);
        unlinked.push(child);
    }
    drop(parent_guard);
    // Free the unlinked nodes after the parent lock is released.
    gone + unlinked.len()
}
