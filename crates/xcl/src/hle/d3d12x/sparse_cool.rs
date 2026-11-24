//! Sparse-window tiles the CPU has not written for a while leave the write-watch look (#582).
//!
//! Before every title submit `title_textures::refresh_once` asks `GetWriteWatch` which pages of
//! the watched ranges were written. That costs about 0.45 ms per GiB on a fast CPU and about
//! 1.4 ms per GiB on a Coffee Lake one, for every committed page whether written or not, and
//! the sparse windows are most of those bytes. A tile (64 KiB) with no written page for
//! `QUIET_LOOKS` looks is made read-only here and the look leaves it out. Its first CPU write
//! faults; the vectored handler makes it writable again and the look takes it back, and the
//! write itself is then seen by the write watch as before.
//!
//! Kernel calls do not fault in the title, they fail (ERROR_NOACCESS), so the places that let
//! the kernel write title memory warm the range first, as they commit lazy pages
//! (`kernelx::lazy::ensure`), and so do the title's own commit, decommit and protect calls.
//!
//! Off unless `XCL_SPARSE_COOL=1`.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    GetWriteWatch, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE,
};

pub(super) const TILE: usize = 0x1_0000;
/// looks in a row with no written page before a tile is made read-only
const QUIET_LOOKS: u64 = 8;
/// tiles made read-only in one look at most, so a look's extra cost stays bounded
const COOL_PER_LOOK: usize = 64;
/// tiles examined in one look at most (a cursor goes round the mapped tiles)
const EXAMINE_PER_LOOK: usize = 1024;
/// `Quiet::last` of a tile this module made read-only (skipped until a write marks it again)
const COOLED: u64 = u64::MAX;
/// looks between rebuilds of the tile list at most: the title maps and unmaps tiles all through a
/// load, and rebuilding the whole list each time cost more than the look it saves. A stale list
/// is harmless: a tile mapped since waits for the next rebuild, one unmapped since fails
/// `cool_tile`'s query.
const REBUILD_EVERY: u64 = 64;

const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// `XCL_SPARSE_COOL=1`
pub fn enabled() -> bool {
    #[cfg(test)]
    if tests::ON.with(|on| on.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    // under Wine the protect and fault round trips cost more than the look saves (#1034)
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_COOL").is_ok_and(|v| v == "1") && !crate::util::under_wine())
}

/// tiles read-only now (left out of the look)
static COOL: RwLock<Option<HashSet<usize>>> = RwLock::new(None);
/// changes when a tile is warmed (at once: the write must be in the next look) and when tiles
/// were cooled (in batches, `GEN_BATCH`): the cooled tiles are taken out of the look's ranges again
static GEN: AtomicU64 = AtomicU64::new(0);
/// tiles cooled, or looks passed with a cooled tile still in the ranges, before `GEN` changes for
/// them: a cooled tile left in the look a while costs a little of the look, filtering the ranges
/// again on every cooling pass cost more (#1034)
const GEN_BATCH: u64 = 256;
const GEN_BATCH_LOOKS: u64 = 64;
/// tiles cooled and warmed so far, and µs spent in `after_look`, for the timing line
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

struct Quiet {
    look: u64,
    /// the mapped tiles with no hole, their index, and where the cursor is in them
    tiles: Vec<usize>,
    index: HashMap<usize, u32>,
    /// per tile: the look it was last seen written (or first seen) in, or `COOLED`
    last: Vec<u64>,
    tiles_gen: u64,
    built_at: u64,
    cursor: usize,
    /// tiles cooled since `GEN` last changed for cooling, and the look it changed in
    unannounced: u64,
    announced_at: u64,
}

static QUIET: Mutex<Option<Quiet>> = Mutex::new(None);

pub fn gen() -> u64 {
    GEN.load(Ordering::Acquire)
}

/// (tiles read-only now, cooled so far, warmed so far, ms spent in the cooling pass)
pub fn counts() -> (usize, u64, u64, f64) {
    let now = COOL.read().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |s| s.len());
    (now, COUNTS[0].load(Ordering::Relaxed), COUNTS[1].load(Ordering::Relaxed), COUNTS[2].load(Ordering::Relaxed) as f64 / 1000.0)
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    // a write (ExceptionInformation[0] == 1) to a read-only tile of ours
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 || r.ExceptionInformation[0] != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let tile = r.ExceptionInformation[1] & !(TILE - 1);
    if warm_tile(tile) || writable(r.ExceptionInformation[1]) {
        // warmed here, or by another thread a moment ago: the write goes on
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

/// `a` can be written now: a tile another thread warmed between this thread's fault and its look
unsafe fn writable(a: usize) -> bool {
    let tile = a & !(TILE - 1);
    if !was_ours(tile) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE
}

/// tiles ever cooled (so the handler only lets writes through to its own tiles)
static EVER: RwLock<Option<HashSet<usize>>> = RwLock::new(None);

fn was_ours(tile: usize) -> bool {
    EVER.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.contains(&tile))
}

/// Make `tile` writable again if it is read-only (true: it was).
fn warm_tile(tile: usize) -> bool {
    let mut g = COOL.write().unwrap_or_else(|e| e.into_inner());
    let Some(set) = g.as_mut() else { return false };
    if !set.remove(&tile) {
        return false;
    }
    // the look's ranges are made again before the tile can be written without a fault: a write
    // between the protect and a later raise would miss the next submit's look
    GEN.fetch_add(1, Ordering::Release);
    let mut old = 0u32;
    // SAFETY: `tile` is a committed 64 KiB tile of a sparse window that this module made
    // read-only; restoring read-write changes no contents. A tile decommitted meanwhile makes
    // this fail, which leaves nothing to restore.
    unsafe { VirtualProtect(tile as *const c_void, TILE, PAGE_READWRITE, &mut old) };
    drop(g);
    COUNTS[1].fetch_add(1, Ordering::Relaxed);
    true
}

/// A tile warmed for a write that does not fault (a kernel call) is not cooled again for this
/// long: an asynchronous write (overlapped `ReadFile`, DirectStorage) lands after the call returns.
const GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Tiles pinned writable for a call in progress (`pin`) and when each was last warmed, kept
/// without a lock: hot paths (`kernelx::lazy::ensure`, the exception emulation) warm on every
/// call. A tile shares its slot with the tiles `SLOTS` tiles (16 GiB) away; a shared slot only
/// keeps a tile writable longer, it never cools one early.
const SLOTS: usize = 1 << 18;
static PINS: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
/// ms since the first warm, plus one (0: never warmed)
static WARMED: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];

fn slot(tile: usize) -> usize {
    (tile / TILE) % SLOTS
}

fn now_ms() -> u32 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    (START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u32).wrapping_add(1).max(1)
}

/// The slots of the tiles of `[p, p + len)`, each once.
fn slots_of(p: usize, len: usize) -> impl Iterator<Item = usize> {
    let first = p / TILE;
    let last = p.saturating_add(len.max(1) - 1) / TILE;
    (0..(last - first + 1).min(SLOTS)).map(move |i| (first + i) % SLOTS)
}

/// Kept from being cooled while it lives (`pin`).
pub struct Pin(usize, usize);

impl Drop for Pin {
    fn drop(&mut self) {
        if self.1 == 0 {
            return;
        }
        note_warmed(self.0, self.1);
        for s in slots_of(self.0, self.1) {
            PINS[s].fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn note_warmed(p: usize, len: usize) {
    let now = now_ms();
    for s in slots_of(p, len) {
        // most warms land on a tile warmed a moment ago: a load costs less than a store there
        if WARMED[s].load(Ordering::Relaxed) != now {
            WARMED[s].store(now, Ordering::Release);
        }
    }
}

/// `[p, p + len)` made writable and kept from being cooled until the returned pin is dropped
/// (and for `GRACE` after): for a kernel call that writes there, which fails instead of faulting.
pub fn pin(p: usize, len: usize) -> Pin {
    if len == 0 || !enabled() {
        return Pin(p, 0);
    }
    // pinned first, then warmed: a cooling pass that protected a tile before the pin is undone
    // by this warm, one after it sees the pin (it looks under the read-only set's write lock,
    // which the warm's read lock waits for)
    for s in slots_of(p, len) {
        PINS[s].fetch_add(1, Ordering::AcqRel);
    }
    warm(p, len);
    Pin(p, len)
}

/// A tile pinned now or warmed less than `GRACE` ago (not cooled).
fn held(tile: usize) -> bool {
    let s = slot(tile);
    if PINS[s].load(Ordering::Acquire) != 0 {
        return true;
    }
    let w = WARMED[s].load(Ordering::Acquire);
    w != 0 && now_ms().wrapping_sub(w) < GRACE.as_millis() as u32
}

/// The read-only tiles of `[p, p + len)` made writable – memory the kernel, XCL or the title's
/// own memory calls are about to write, commit, decommit or protect – and not cooled again for
/// `GRACE`.
pub fn warm(p: usize, len: usize) {
    if len == 0 || !enabled() {
        return;
    }
    note_warmed(p, len);
    let tiles: Vec<usize> = {
        let g = COOL.read().unwrap_or_else(|e| e.into_inner());
        let Some(set) = g.as_ref() else { return };
        if set.is_empty() {
            return;
        }
        let first = p & !(TILE - 1);
        let last = p.saturating_add(len - 1) & !(TILE - 1);
        if (last - first) / TILE < set.len() {
            (0..=(last - first) / TILE).map(|i| first + i * TILE).filter(|t| set.contains(t)).collect()
        } else {
            set.iter().copied().filter(|&t| t >= first && t <= last).collect()
        }
    };
    for t in tiles {
        warm_tile(t);
    }
}

/// A cooled tile's protection as the title set it (read-write): for `XMemVirtualQuery`, so the
/// title never sees this module's read-only pages.
pub fn as_title_sees(base: usize, protect: u32) -> u32 {
    if protect == PAGE_READONLY && enabled() && COOL.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.contains(&(base & !(TILE - 1)))) {
        PAGE_READWRITE
    } else {
        protect
    }
}

#[cfg(test)]
pub(crate) fn enable_on_this_thread_for_test() {
    tests::ON.with(|on| on.set(true));
}

#[cfg(test)]
pub(crate) unsafe fn cool_for_test(tile: usize) -> bool {
    // a test beside this one may have warmed the same address before freeing it (its grace)
    WARMED[slot(tile)].store(0, Ordering::Release);
    cool_tile(tile)
}

/// Read-only tiles in a row the look leaves out at least: each cut is one more `GetWriteWatch`
/// call (about 1 µs), a tile left in costs about 0.09 µs on a Coffee Lake (1.4 ms per GiB) and
/// less on a newer CPU, so a cut pays from about a dozen tiles. A read-only tile left in the look
/// is still warmed by its first write.
const MIN_CUT: usize = 16;

/// `runs` (`[start, end)`) without the runs of at least `MIN_CUT` read-only tiles.
pub fn without_cool(runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    without_cooled_runs(runs, MIN_CUT)
}

/// Whether `tile` is read-only now.
#[cfg(test)]
pub(crate) fn is_cooled(tile: usize) -> bool {
    COOL.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.contains(&tile))
}

fn without_cooled_runs(runs: Vec<(usize, usize)>, min_tiles: usize) -> Vec<(usize, usize)> {
    if !enabled() {
        return runs;
    }
    // the cooled tiles sorted and joined into spans, then cut out of each run: the runs cover
    // gigabytes, the cooled set is thousands of tiles
    let mut cool: Vec<usize> = {
        let g = COOL.read().unwrap_or_else(|e| e.into_inner());
        match g.as_ref() {
            Some(s) if !s.is_empty() => s.iter().copied().collect(),
            _ => return runs,
        }
    };
    cool.sort_unstable();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < cool.len() {
        let mut j = i + 1;
        while j < cool.len() && cool[j] == cool[j - 1] + TILE {
            j += 1;
        }
        if j - i >= min_tiles {
            spans.push((cool[i], cool[j - 1] + TILE));
        }
        i = j;
    }
    let mut out = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        let mut start = a;
        let first = spans.partition_point(|&(_, e)| e <= a);
        for &(s, e) in spans[first..].iter().take_while(|&&(s, _)| s < b) {
            if start < s {
                out.push((start, s));
            }
            start = start.max(e);
        }
        if start < b {
            out.push((start, b));
        }
    }
    out
}

/// After a look: `dirty` (the written pages, sorted) mark their tiles written; then up to
/// `EXAMINE_PER_LOOK` of `tiles()` (the mapped tiles with no hole, made again when `tiles_gen`
/// changes) are examined and quiet ones made read-only.
pub unsafe fn after_look(dirty: &[usize], tiles_gen: u64, tiles: impl FnOnce() -> Vec<usize>) {
    if !enabled() {
        return;
    }
    let t0 = std::time::Instant::now();
    after_look_inner(dirty, tiles_gen, tiles);
    COUNTS[2].fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
}

unsafe fn after_look_inner(dirty: &[usize], tiles_gen: u64, tiles: impl FnOnce() -> Vec<usize>) {
    let mut g = QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let q = g.get_or_insert_with(|| Quiet {
        look: 0,
        tiles: Vec::new(),
        index: HashMap::new(),
        last: Vec::new(),
        tiles_gen: u64::MAX,
        built_at: 0,
        cursor: 0,
        unannounced: 0,
        announced_at: 0,
    });
    q.look += 1;
    let look = q.look;
    if q.tiles_gen != tiles_gen && (q.tiles.is_empty() || look - q.built_at >= REBUILD_EVERY) {
        let tiles = tiles();
        let last: Vec<u64> = tiles.iter().map(|t| q.index.get(t).map_or(look, |&i| q.last[i as usize])).collect();
        q.index = tiles.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        q.tiles = tiles;
        q.last = last;
        q.tiles_gen = tiles_gen;
        q.built_at = look;
        q.cursor = 0;
    }
    // a written tile (a cooled one too: its write faulted and warmed it) is quiet again from now
    let mut prev = usize::MAX;
    for &p in dirty {
        let t = p & !(TILE - 1);
        if t != prev {
            if let Some(&i) = q.index.get(&t) {
                q.last[i as usize] = look;
            }
            prev = t;
        }
    }
    if q.tiles.is_empty() {
        return;
    }
    let mut cooled = 0usize;
    for _ in 0..q.tiles.len().min(EXAMINE_PER_LOOK) {
        if cooled == COOL_PER_LOOK {
            break;
        }
        q.cursor = (q.cursor + 1) % q.tiles.len();
        let i = q.cursor;
        let seen = q.last[i];
        if seen == COOLED || look - seen < QUIET_LOOKS {
            continue;
        }
        if cool_tile(q.tiles[i]) {
            q.last[i] = COOLED;
            cooled += 1;
        } else {
            q.last[i] = look;
        }
    }
    if cooled > 0 {
        COUNTS[0].fetch_add(cooled as u64, Ordering::Relaxed);
        q.unannounced += cooled as u64;
    }
    announce(q);
}

fn announce(q: &mut Quiet) {
    if q.unannounced > 0 && (q.unannounced >= GEN_BATCH || q.look - q.announced_at >= GEN_BATCH_LOOKS) {
        GEN.fetch_add(1, Ordering::Release);
        q.unannounced = 0;
        q.announced_at = q.look;
    }
}

/// Make `tile` read-only if it is all committed read-write and not written since the last look
/// (true: it is read-only now).
unsafe fn cool_tile(tile: usize) -> bool {
    {
        let g = COOL.read().unwrap_or_else(|e| e.into_inner());
        if g.as_ref().is_some_and(|s| s.contains(&tile)) {
            return false;
        }
    }
    // cheap first: a held tile is passed over without a query (looked at again under the lock)
    if held(tile) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(tile as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0
        || m.State != MEM_COMMIT
        || m.Protect != PAGE_READWRITE
        || (m.BaseAddress as usize).saturating_add(m.RegionSize) < tile + TILE
    {
        return false;
    }
    install();
    EVER.write().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(tile);
    let mut g = COOL.write().unwrap_or_else(|e| e.into_inner());
    if held(tile) {
        return false;
    }
    let mut old = 0u32;
    if VirtualProtect(tile as *const c_void, TILE, PAGE_READONLY, &mut old) == 0 {
        return false;
    }
    // A write between the look and the protect is in the write watch but the look will leave
    // this tile out from now on: such a tile stays writable (not reset, so the next look sees it).
    let mut buf = [std::ptr::null_mut::<c_void>(); 1];
    let mut count = buf.len();
    let mut gran = 0u32;
    if GetWriteWatch(0, tile as *const c_void, TILE, buf.as_mut_ptr(), &mut count, &mut gran) != 0 || count != 0 {
        VirtualProtect(tile as *const c_void, TILE, old, &mut old);
        return false;
    }
    g.get_or_insert_with(HashSet::new).insert(tile);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE};

    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;

    thread_local! {
        /// `enabled()` on this test's thread only (other tests run beside it with it off)
        pub(super) static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    unsafe fn watched(tiles: usize) -> usize {
        ON.with(|on| on.set(true));
        let p = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
        assert_ne!(p, 0);
        // an earlier test may have warmed the same addresses (their grace would keep these warm)
        for s in slots_of(p, tiles * TILE) {
            WARMED[s].store(0, Ordering::Release);
        }
        p
    }

    unsafe fn written(base: usize, len: usize) -> usize {
        let mut buf = vec![std::ptr::null_mut::<c_void>(); len / 0x1000];
        let mut n = buf.len();
        let mut gran = 0u32;
        assert_eq!(GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, len, buf.as_mut_ptr(), &mut n, &mut gran), 0);
        n
    }

    unsafe fn protect(a: usize) -> u32 {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m));
        m.Protect
    }

    /// A cooled tile leaves the runs; its first write faults, warms it and is still seen by the
    /// write watch afterwards.
    #[test]
    fn a_cooled_tile_is_warmed_by_its_first_write_and_the_write_is_watched() {
        unsafe {
            let p = watched(4);
            written(p, 4 * TILE);
            assert!(cool_tile(p + TILE));
            assert_eq!(protect(p + TILE), PAGE_READONLY);
            assert_eq!(without_cooled_runs(vec![(p, p + 4 * TILE)], 1), vec![(p, p + TILE), (p + 2 * TILE, p + 4 * TILE)]);
            std::ptr::write_volatile((p + TILE + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((p + TILE + 8) as *const u32), 7);
            assert_eq!(protect(p + TILE), PAGE_READWRITE);
            assert_eq!(without_cooled_runs(vec![(p, p + 4 * TILE)], 1), vec![(p, p + 4 * TILE)]);
            assert_eq!(written(p + TILE, TILE), 1, "the write after the fault is in the write watch");
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A tile written after the look and before the protect is not cooled (the look would miss it).
    #[test]
    fn a_tile_written_since_the_look_is_not_cooled() {
        unsafe {
            let p = watched(1);
            written(p, TILE);
            std::ptr::write_volatile(p as *mut u32, 1);
            assert!(!cool_tile(p));
            assert_eq!(protect(p), PAGE_READWRITE);
            assert_eq!(written(p, TILE), 1, "still reported to the next look");
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `warm` makes a range's read-only tiles writable without a fault (kernel writes).
    #[test]
    fn warm_restores_write_access_for_kernel_writes() {
        unsafe {
            let p = watched(3);
            written(p, 3 * TILE);
            assert!(cool_tile(p));
            assert!(cool_tile(p + TILE));
            assert!(cool_tile(p + 2 * TILE));
            warm(p + 100, TILE);
            assert_eq!(protect(p), PAGE_READWRITE);
            assert_eq!(protect(p + TILE), PAGE_READWRITE);
            assert_eq!(protect(p + 2 * TILE), PAGE_READONLY, "outside the range");
            warm(p + 2 * TILE, 1);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A pinned tile is not cooled while the pin lives nor for `GRACE` after (a kernel write in
    /// progress), and a warm alone also keeps it warm for `GRACE`.
    #[test]
    fn a_pinned_or_just_warmed_tile_is_not_cooled() {
        unsafe {
            let p = watched(2);
            written(p, 2 * TILE);
            let pin = pin(p, 16);
            assert!(!cool_tile(p), "pinned");
            drop(pin);
            assert!(!cool_tile(p), "within the grace after the pin");
            warm(p + TILE, 1);
            assert!(!cool_tile(p + TILE), "within the grace after a warm");
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A pin taken after a tile was cooled warms it at once.
    #[test]
    fn a_pin_warms_a_cooled_tile() {
        unsafe {
            let p = watched(1);
            written(p, TILE);
            assert!(cool_tile(p));
            assert_eq!(protect(p), PAGE_READONLY);
            assert_eq!(as_title_sees(p, PAGE_READONLY), PAGE_READWRITE, "the title sees its own protection");
            let _pin = pin(p + 4, 4);
            assert_eq!(protect(p), PAGE_READWRITE);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// The look leaves out a run of `MIN_CUT` read-only tiles, not a shorter one: a cut costs a
    /// `GetWriteWatch` call more.
    #[test]
    fn only_runs_of_min_cut_read_only_tiles_leave_the_look() {
        unsafe {
            let n = MIN_CUT + 3;
            let p = watched(n);
            written(p, n * TILE);
            for i in 1..MIN_CUT {
                assert!(cool_tile(p + i * TILE));
            }
            assert_eq!(without_cool(vec![(p, p + n * TILE)]), vec![(p, p + n * TILE)], "{} tiles in a row", MIN_CUT - 1);
            assert!(cool_tile(p + MIN_CUT * TILE));
            assert_eq!(without_cool(vec![(p, p + n * TILE)]), vec![(p, p + TILE), (p + (MIN_CUT + 1) * TILE, p + n * TILE)]);
            warm(p, n * TILE);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A tile's cut is exact: a run starting or ending inside a read-only tile loses that tile's part.
    #[test]
    fn the_ranges_lose_exactly_the_cooled_tiles() {
        unsafe {
            let p = watched(4);
            written(p, 4 * TILE);
            assert!(cool_tile(p + TILE));
            assert!(cool_tile(p + 3 * TILE));
            // a run from inside tile 0 to inside tile 3, and one inside tile 1 only
            assert_eq!(without_cooled_runs(vec![(p + 0x3000, p + 3 * TILE + 0x2000)], 1), vec![(p + 0x3000, p + TILE), (p + 2 * TILE, p + 3 * TILE)]);
            assert_eq!(without_cooled_runs(vec![(p + TILE + 0x1000, p + TILE + 0x2000)], 1), vec![]);
            warm(p, 4 * TILE);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Quiet tiles are cooled after `QUIET_LOOKS` looks, a tile written every look is not.
    #[test]
    fn only_tiles_quiet_for_several_looks_are_cooled() {
        unsafe {
            let p = watched(2);
            written(p, 2 * TILE);
            // a generation of its own so this test's tile list is used
            let tiles_gen = p as u64;
            for _ in 0..QUIET_LOOKS * 2 {
                std::ptr::write_volatile(p as *mut u32, 1);
                assert!(written(p, 2 * TILE) >= 1);
                after_look(&[p], tiles_gen, || vec![p, p + TILE]);
            }
            assert_eq!(protect(p), PAGE_READWRITE, "written every look");
            assert_eq!(protect(p + TILE), PAGE_READONLY, "never written");
            warm(p, 2 * TILE);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
