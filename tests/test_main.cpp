// deltaforge_test — every stage check, re-run by one command (`ctest`).
// Exit code = number of failed checks, so CTest goes red on any failure.
// The loader checks open the HF oracle dumps; the arena checks need a GPU
// (cudaMalloc), so this binary no longer runs on a machine without one.

#include "safetensors.hpp"
#include "manifest.hpp"
#include "arena.hpp"
#include "bf16.hpp"
#include "compare.hpp"
#include "kernels.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int failures = 0;

const std::wstring MODEL_NAME = L"Qwen3.5-2B";

// OK / FAIL for things that must hold; XFAIL / BAD for negative controls (decision 9).
void check(const std::string& label, bool ok, bool expect = true) {
    const char* status = ok == expect ? (expect ? "OK   " : "XFAIL") : (expect ? "FAIL " : "BAD  ");
    if (ok != expect) ++failures;
    std::cout << status << ' ' << label << '\n';
}

std::wstring dumpPath(const std::wstring& name) {
    return std::wstring(DELTAFORGE_ROOT) + L"/dumps/" + name;
}

std::wstring modelPath(const std::wstring& name) {
    return std::wstring(DELTAFORGE_ROOT) + L"/models/" + name;
}

bool hasShape(const TensorView& tv, std::initializer_list<int64_t> dims, DType dtype) {
    if (tv.dtype != dtype || tv.ndim != static_cast<int>(dims.size())) return false;
    int i = 0;
    for (int64_t d : dims) if (tv.shape[i++] != d) return false;
    return true;
}

// The bf16 dump: 196 taps written by python/dump_oracle.py (brief 04, task 1).
void checkBf16Dump() {
    SafeTensors st(dumpPath(L"oracle_bf16.safetensors"));
    st.loadTensorsToViews();

    check("bf16: 196 tensors", st.getTensors().size() == 196);
    check("bf16: ids is I64 [30]", hasShape(st.at("ids"), {30}, DType::I64));
    check("bf16: logits is BF16 [30, 248320]", hasShape(st.at("logits"), {30, 248320}, DType::BF16));
    check("bf16: embed is BF16 [30, 2048]", hasShape(st.at("embed"), {30, 2048}, DType::BF16));
    // GDN recurrent state stays F32 even in a bf16 run (HF never casts it back)
    check("bf16: L00.cache.recurrent is F32 [16, 128, 128]",
          hasShape(st.at("L00.cache.recurrent"), {16, 128, 128}, DType::F32));
    check("bf16: prompt metadata present", !st.metadata().at("prompt").get<std::string>().empty());
    check("bf16: dtype metadata is bf16", st.metadata().at("dtype").get<std::string>() == "bf16");
    // the GDN math the engine reimplements must be the torch fallback, not an fla kernel
    check("bf16: gdn_impl is the torch fallback",
          st.metadata().at("gdn_impl").get<std::string>().find("modeling_qwen3_5") != std::string::npos);

    // device pointers stay null until something uploads and binds (NFR: no GPU in the loader)
    bool any_device_ptr = false;
    for (const auto& [name, tv] : st.getTensors()) any_device_ptr |= (tv.device_ptr != nullptr);
    check("bf16: device pointers null before bindToDevice", any_device_ptr, false);

    // names come back sorted, so manifests can be diffed against Python's
    std::vector<std::string> names = st.sortedNames();
    check("bf16: sortedNames is sorted", std::is_sorted(names.begin(), names.end()));

    // negative control: an unknown name must throw, not return garbage
    bool threw = false;
    try { st.at("no.such.tensor"); } catch (const std::exception&) { threw = true; }
    check("bf16: unknown name throws", threw);
}

// The fp32 anchor dump: same taps, every float tensor widened.
void checkFp32Dump() {
    SafeTensors st(dumpPath(L"oracle_fp32.safetensors"));
    st.loadTensorsToViews();

    check("fp32: 196 tensors", st.getTensors().size() == 196);
    check("fp32: dtype metadata is fp32", st.metadata().at("dtype").get<std::string>() == "fp32");
    check("fp32: logits is F32 [30, 248320]", hasShape(st.at("logits"), {30, 248320}, DType::F32));
    check("fp32: ids still I64 [30]", hasShape(st.at("ids"), {30}, DType::I64));

    size_t bf16_count = 0;
    for (const auto& [name, tv] : st.getTensors()) bf16_count += (tv.dtype == DType::BF16);
    check("fp32: no BF16 tensors", bf16_count == 0);
}

// ── bf16 helpers ────────────────────────────────────────────────────────────
// Read element i of an F32 TensorView. memcpy, not a float* cast: the mmap blob
// carries no alignment guarantee for an arbitrary tensor offset.
float f32At(const TensorView& tv, size_t i) {
    float f;
    std::memcpy(&f, tv.host_ptr + i * sizeof(float), sizeof(float));
    return f;
}

bool isNanPattern(uint16_t bits) {
    return (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0;
}

// What a truncating implementation would produce — the classic mistake, kept as
// a negative control so the table is proven able to catch it.
uint16_t truncate_to_bf16(float f) {
    return static_cast<uint16_t>(f32_bits(f) >> 16);
}

// torch is the oracle (dumps/bf16_table.safetensors, written by gen_bf16_table.py).
void checkBf16Helpers() {
    SafeTensors st(dumpPath(L"bf16_table.safetensors"));
    st.loadTensorsToViews();

    const TensorView& widen = st.at("widen");
    const TensorView& cast_in = st.at("cast_in");
    const TensorView& cast_out = st.at("cast_out");
    check("bf16: table is F32[65536]", hasShape(widen, {65536}, DType::F32));
    check("bf16: cast_in and cast_out agree in length", cast_in.shape[0] == cast_out.shape[0]);

    // 1. widening is exact for every one of the 65,536 patterns (bit compare, so
    //    the 254 NaN patterns are checked rather than skipped by NaN != NaN)
    size_t widen_bad = 0;
    uint16_t widen_first = 0;
    for (uint32_t i = 0; i < 65536; ++i) {
        const uint16_t bits = static_cast<uint16_t>(i);
        if (f32_bits(bf16_to_f32(bits)) != f32_bits(f32At(widen, i))) {
            if (widen_bad == 0) widen_first = bits;
            ++widen_bad;
        }
    }
    check("bf16: bf16_to_f32 matches torch on all 65536 patterns", widen_bad == 0);
    if (widen_bad) std::cout << "       first mismatch at pattern 0x" << std::hex << widen_first << std::dec
                             << ", " << widen_bad << " total\n";

    // 2. round-trip, no oracle needed: narrowing an exactly-representable value
    //    must be the identity. NaN is the one exception -- all 254 patterns
    //    canonicalise to torch's 0xFFFF.
    size_t rt_bad = 0, nan_bad = 0;
    for (uint32_t i = 0; i < 65536; ++i) {
        const uint16_t bits = static_cast<uint16_t>(i);
        const uint16_t again = f32_to_bf16(bf16_to_f32(bits));
        if (isNanPattern(bits)) { nan_bad += (again != 0xFFFF); }
        else { rt_bad += (again != bits); }
    }
    check("bf16: round-trip is the identity for every finite pattern", rt_bad == 0);
    check("bf16: every NaN pattern narrows to torch's 0xFFFF", nan_bad == 0);

    // 3. narrowing vs torch on the edge cases + 12k random values
    const size_t n = static_cast<size_t>(cast_in.shape[0]);
    size_t cast_bad = 0, trunc_agrees = 0;
    size_t first_bad = 0;
    for (size_t j = 0; j < n; ++j) {
        const float in = f32At(cast_in, j);
        const uint32_t want = f32_bits(f32At(cast_out, j));
        if (bf16_widened_bits(f32_to_bf16(in)) != want) {
            if (cast_bad == 0) first_bad = j;
            ++cast_bad;
        }
        trunc_agrees += (bf16_widened_bits(truncate_to_bf16(in)) == want);
    }
    check("bf16: f32_to_bf16 matches torch on every test value", cast_bad == 0);
    if (cast_bad) std::cout << "       first mismatch at cast_in[" << first_bad << "] = "
                            << f32At(cast_in, first_bad) << ", " << cast_bad << " of " << n << "\n";

    // Negative control: truncation must NOT match torch, or the table above
    // cannot tell round-to-nearest-even from the lazy implementation.
    std::cout << "       truncation agrees on " << trunc_agrees << " of " << n << " values\n";
    check("bf16: control — truncation disagrees with torch", trunc_agrees == n, false);
}

// ── arena ───────────────────────────────────────────────────────────────────
// Did `expr` throw? Used for the "must throw" checks, so one bad call can't
// take the whole run down with it.
template <class F>
bool threw(F&& expr) {
    try { expr(); return false; } catch (const std::exception&) { return true; }
}

// One realistic decoder step's temporaries: T=30 tokens, hidden 2048, 8 query
// heads and 2 kv heads of 256, MLP intermediate 6144. ~1.2 MB per call.
void allocStepTensors(Arena& a) {
    a.allocTensor(DType::BF16, {30, 2048});      // input_norm
    a.allocTensor(DType::BF16, {30, 8, 256});    // q
    a.allocTensor(DType::BF16, {30, 2, 256});    // k
    a.allocTensor(DType::BF16, {30, 2, 256});    // v
    a.allocTensor(DType::F32,  {8, 30, 30});     // attention scores (fp32)
    a.allocTensor(DType::BF16, {30, 2048});      // gate
    a.allocTensor(DType::BF16, {30, 6144});      // mlp gate_proj
    a.allocTensor(DType::BF16, {30, 6144});      // mlp up_proj
    a.allocTensor(DType::BF16, {30, 2048});      // mixer_out
}

void checkArena() {
    Arena a(16 << 20);
    check("arena: starts empty", a.used() == 0 && a.highWater() == 0);
    check("arena: capacity is what we asked for", a.capacity() == (16u << 20));

    // bump allocation: 256-aligned handouts, used() tracks the padding
    char* p1 = static_cast<char*>(a.alloc(1));
    char* p2 = static_cast<char*>(a.alloc(1));
    check("arena: handouts are 256-aligned",
          reinterpret_cast<uintptr_t>(p1) % 256 == 0 && reinterpret_cast<uintptr_t>(p2) % 256 == 0);
    check("arena: second handout is one alignment on", p2 - p1 == 256);
    check("arena: used() counts alignment padding", a.used() == 257);

    // a failed request must leave the arena exactly as it was
    size_t before_used = a.used();
    check("arena: over-capacity request throws", threw([&] { a.alloc(a.capacity()); }));
    check("arena: alignment must be a power of two", threw([&] { a.alloc(8, 3); }));
    check("arena: a failed request does not move the offset", a.used() == before_used);

    // reset() rewinds the offset and frees nothing; highWater() is the real ceiling
    a.reset();
    check("arena: reset() zeroes used()", a.used() == 0);
    check("arena: reset() keeps highWater()", a.highWater() == 257);
    check("arena: reset() hands back the base pointer", static_cast<char*>(a.alloc(1)) == p1);

    // shape validation happens before any allocation (see to_shape / nbytes_of)
    a.reset();
    before_used = a.used();
    check("arena: ndim > MAX_DIM throws", threw([&] { a.allocTensor(DType::F32, {1, 2, 3, 4, 5, 6}); }));
    check("arena: zero dim throws", threw([&] { a.allocTensor(DType::F32, {0}); }));
    check("arena: negative dim throws", threw([&] { a.allocTensor(DType::F32, {-4}); }));
    check("arena: tensor larger than the arena throws",
          threw([&] { a.allocTensor(DType::BF16, {248320, 2048}); }));
    check("arena: a failed allocTensor does not move the offset", a.used() == before_used);

    // allocTensor shape/size bookkeeping
    a.reset();
    Tensor t = a.allocTensor(DType::BF16, {30, 2048});
    check("arena: allocTensor shape survives", t.ndim == 2 && t.dim(0) == 30 && t.dim(1) == 2048);
    check("arena: allocTensor nbytes is numel * elem_size", t.nbytes() == 30ull * 2048 * 2);
    check("arena: allocTensor consumed exactly that", a.used() == t.nbytes());
    Tensor u = a.allocTensor(DType::F32, {7});
    check("arena: successive tensors do not overlap",
          static_cast<char*>(u.data) - static_cast<char*>(t.data) >= static_cast<ptrdiff_t>(t.nbytes()));

    // fromView: a null device_ptr means the weight was never uploaded
    TensorView unbound{};
    unbound.ndim = 1;
    unbound.shape = {2048, 0, 0, 0, 0};
    unbound.dtype = DType::BF16;
    check("arena: fromView on an unbound view throws", threw([&] { fromView(unbound); }));
}

// The alloc/free counters are the leak detector every later check leans on, so
// they get verified themselves: only a real cudaMalloc/cudaFree may move them.
void checkDeviceBufferCounters() {
    const size_t allocs = DeviceBuffer::getAllocCount();
    const size_t frees = DeviceBuffer::getFreeCount();

    { DeviceBuffer one(1024); }
    check("buffer: one buffer = one alloc, one free",
          DeviceBuffer::getAllocCount() == allocs + 1 && DeviceBuffer::getFreeCount() == frees + 1);

    // an empty buffer owns nothing, so neither counter may move
    const size_t a1 = DeviceBuffer::getAllocCount();
    const size_t f1 = DeviceBuffer::getFreeCount();
    { DeviceBuffer empty; }
    check("buffer: a default-constructed buffer frees nothing",
          DeviceBuffer::getAllocCount() == a1 && DeviceBuffer::getFreeCount() == f1);

    // moving transfers ownership: one alloc, and exactly one free at the end
    const size_t a2 = DeviceBuffer::getAllocCount();
    const size_t f2 = DeviceBuffer::getFreeCount();
    {
        DeviceBuffer target;
        target = DeviceBuffer(1024);          // move-assign over an empty buffer
        DeviceBuffer moved(std::move(target));  // move-construct: still one owner
    }
    check("buffer: a moved buffer is freed once, not twice",
          DeviceBuffer::getAllocCount() == a2 + 1 && DeviceBuffer::getFreeCount() == f2 + 1);

    // the invariant that makes the counters a leak detector at all
    check("buffer: every buffer in this scope is accounted for",
          DeviceBuffer::getFreeCount() - frees == DeviceBuffer::getAllocCount() - allocs);
}

// The point of the arena: 10k reset cycles must not touch cudaMalloc even once.
void checkArenaReuse() {
    Arena a(16 << 20);

    allocStepTensors(a);
    const size_t step_high_water = a.highWater();
    a.reset();

    const size_t allocs_before = DeviceBuffer::getAllocCount();
    bool high_water_stable = true;
    for (int i = 0; i < 10000; ++i) {
        allocStepTensors(a);
        high_water_stable &= (a.highWater() == step_high_water);
        a.reset();
    }
    const size_t allocs_after = DeviceBuffer::getAllocCount();

    std::cout << "       step high-water: " << step_high_water << " B of " << a.capacity() << " B\n";
    check("arena: 10k reset cycles cause zero device allocations", allocs_after == allocs_before);
    check("arena: highWater() is stable across cycles", high_water_stable);
    check("arena: used() is 0 after the last reset", a.used() == 0);

    // Negative control: the same loop with a real device allocation in it MUST
    // move the counter. Without this, the check above also passes when the
    // counter is simply never incremented.
    const size_t control_before = DeviceBuffer::getAllocCount();
    for (int i = 0; i < 10; ++i) {
        DeviceBuffer scratch(1024);
        allocStepTensors(a);
        a.reset();
    }
    check("arena: control — a DeviceBuffer in the loop does allocate",
          DeviceBuffer::getAllocCount() == control_before, false);
}

// ── compare ─────────────────────────────────────────────────────────────────
// Every stage check prints its numbers whether it passes or fails: task 7 has to
// *record* a calibrated tolerance, and you can only record what you printed.
void report(const CompareResult& r) {
    std::cout << "       n=" << r.n
              << "  max|d|=" << std::scientific << std::setprecision(3) << r.max_abs_diff
              << "  max|ref|=" << r.max_ref
              << (r.rel_is_absolute ? "  abs=" : "  rel=") << r.rel
              << std::defaultfloat;
    if (r.max_abs_diff > 0) std::cout << "  worst[" << r.worst << "] " << r.worst_mine << " vs " << r.worst_ref;
    if (r.nan_both) std::cout << "  nan_both=" << r.nan_both;
    if (r.nan_one) std::cout << "  NAN_ONE_SIDE=" << r.nan_one;
    if (r.inf_mismatch) std::cout << "  INF_MISMATCH=" << r.inf_mismatch;
    std::cout << '\n';
}

void compare(const std::string& label, const Context& ctx, const Tensor& mine,
             const TensorView& ref, double rel_tol) {
    const CompareResult r = compareToRef(ctx, mine, ref);
    report(r);
    check(label, r.within(rel_tol));
}

// Negative control: these MUST differ. XFAIL when they do, BAD when they don't.
void expect_differs(const std::string& label, const Context& ctx, const Tensor& mine,
                    const TensorView& ref, double rel_tol) {
    const CompareResult r = compareToRef(ctx, mine, ref);
    report(r);
    check(label, r.within(rel_tol), false);
}

// Upload a dump tensor into the arena: the first thing in this file that writes
// device memory (until now the arena only handed out addresses).
Tensor uploadView(Arena& a, const TensorView& ref) {
    void* p = a.alloc(ref.nbytes);
    CUDA_CHECK(cudaMemcpy(p, ref.host_ptr, ref.nbytes, cudaMemcpyHostToDevice));
    return Tensor(p, ref.dtype, ref.shape, ref.ndim);
}

void checkCompare() {
    SafeTensors bf16_dump(dumpPath(L"oracle_bf16.safetensors"));
    bf16_dump.loadTensorsToViews();
    SafeTensors fp32_dump(dumpPath(L"oracle_fp32.safetensors"));
    fp32_dump.loadTensorsToViews();

    Arena a(16 << 20);
    Context ctx(a);

    const TensorView& ref_bf16 = bf16_dump.at("final_norm");   // BF16 [30, 2048]
    const TensorView& ref_fp32 = fp32_dump.at("final_norm");   // F32  [30, 2048]

    // 1. self-compare: the same bytes, round-tripped through the device. Any
    //    non-zero result here is a widening or indexing bug, not rounding.
    Tensor mine = uploadView(a, ref_bf16);
    compare("compare: self-compare is exact", ctx, mine, ref_bf16, 0.0);

    // 2. cross-dtype, and a real measurement: the same tap from the bf16 run vs
    //    the fp32 run. This is the bf16 activation error, printed for the record.
    compare("compare: bf16 run vs fp32 anchor within 1e-2", ctx, mine, ref_fp32, 1e-2);

    // 3. one perturbed element must be caught — and located
    const size_t victim = 777;
    uint16_t orig = 0;
    std::memcpy(&orig, ref_bf16.host_ptr + victim * 2, 2);
    const uint16_t bumped = f32_to_bf16(bf16_to_f32(orig) * 1.5f + 1.0f);
    CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(mine.data) + victim * 2, &bumped, 2, cudaMemcpyHostToDevice));

    const CompareResult perturbed = compareToRef(ctx, mine, ref_bf16);
    report(perturbed);
    check("compare: a one-element perturbation is detected", !perturbed.within(0.0));
    check("compare: the perturbed element is located", perturbed.worst == victim);

    // 4. the same perturbation through the negative-control path
    expect_differs("compare: control — perturbed tensor differs", ctx, mine, ref_bf16, 0.0);

    // mismatched sizes are a hard error, not a tolerance question
    // "embed" would NOT do: it is [30, 2048] too, the same 61440 elements
    bool size_threw = threw([&] { compareToRef(ctx, mine, bf16_dump.at("L00.cache.recurrent")); });
    check("compare: mismatched element counts throw", size_threw);

    // ids are integers: exact or wrong, never "close"
    Tensor ids = uploadView(a, bf16_dump.at("ids"));
    check("compare: I64 ids compare exactly", compareExactI64(ctx, ids, bf16_dump.at("ids")) == 0);
    check("compare: a float compare on I64 throws",
          threw([&] { compareToRef(ctx, ids, bf16_dump.at("ids")); }));
}

// ── kernel plumbing ─────────────────────────────────────────────────────────
// Everything a kernel launch needs, proven BEFORE any kernel body exists: the
// launch config, the stream, the host-side shape guards, and the error path. No
// model file and no dump -- a tiny synthetic table is enough, and keeps each
// later failure attributable to the kernel body alone.

// A TensorView over host memory, so compare() can grade against an expected
// buffer we computed ourselves rather than a dump.
TensorView hostView(const void* data, DType dtype, std::initializer_list<int64_t> dims) {
    auto [shape, ndim] = to_shape(dims);
    TensorView tv{};
    tv.host_ptr = static_cast<const uint8_t*>(data);
    tv.device_ptr = nullptr;
    tv.shape = shape;
    tv.ndim = ndim;
    tv.dtype = dtype;
    tv.nbytes = nbytes_of(dtype, shape, ndim);
    return tv;
}

Tensor uploadBytes(Arena& a, const void* src, DType dtype, std::initializer_list<int64_t> dims) {
    Tensor t = a.allocTensor(dtype, dims);
    CUDA_CHECK(cudaMemcpy(t.data, src, t.nbytes(), cudaMemcpyHostToDevice));
    return t;
}

void checkKernelPlumbing() {
    Arena a(16 << 20);
    Context ctx(a);

    const int64_t V = 8, H = 2048, T = 4;

    // Distinct raw bit patterns, not floats: the gather is a 2-byte copy, so the
    // table's values only need to be unique per element. 0..V*H stays below
    // 0x4000, so none of them is a NaN or an inf that compare() would special-case.
    std::vector<uint16_t> table_host(static_cast<size_t>(V * H));
    for (int64_t r = 0; r < V; ++r)
        for (int64_t c = 0; c < H; ++c)
            table_host[static_cast<size_t>(r * H + c)] = static_cast<uint16_t>(r * H + c);

    const std::vector<int64_t> ids_host = {5, 0, 7, 2};
    std::vector<uint16_t> want(static_cast<size_t>(T * H));
    for (int64_t t = 0; t < T; ++t)
        std::memcpy(&want[static_cast<size_t>(t * H)], &table_host[static_cast<size_t>(ids_host[t] * H)],
                    static_cast<size_t>(H) * 2);

    Tensor table = uploadBytes(a, table_host.data(), DType::BF16, {V, H});
    Tensor ids = uploadBytes(a, ids_host.data(), DType::I64, {T});
    Tensor out = a.allocTensor(DType::BF16, {T, H});
    const TensorView want_view = hostView(want.data(), DType::BF16, {T, H});

    // 1. the launch itself: grid/block arithmetic, the stream, the error path
    embedGather(ctx, out, table, ids);
    ctx.sync();
    check("kernel: embedGather launches cleanly", cudaGetLastError() == cudaSuccess);

    Tensor weight = a.allocTensor(DType::BF16, {H});
    CUDA_CHECK(cudaMemset(weight.data, 0, weight.nbytes()));
    Tensor norm_out = a.allocTensor(DType::BF16, {T, H});
    rmsNorm(ctx, norm_out, out, weight, 1e-6f);
    ctx.sync();
    check("kernel: rmsNorm launches cleanly", cudaGetLastError() == cudaSuccess);

    // 2. the host-side guards. A shape mismatch between two stages is a far more
    //    common bug than wrong arithmetic, and these are what catch it.
    Tensor out_f32 = a.allocTensor(DType::F32, {T, H});
    Tensor out_wrong_rows = a.allocTensor(DType::BF16, {T + 1, H});
    Tensor out_1d = a.allocTensor(DType::BF16, {T * H});
    Tensor ids_f32 = a.allocTensor(DType::F32, {T});
    check("kernel: gather rejects a non-BF16 out", threw([&] { embedGather(ctx, out_f32, table, ids); }));
    check("kernel: gather rejects non-I64 ids", threw([&] { embedGather(ctx, out, table, ids_f32); }));
    check("kernel: gather rejects a 1-D out", threw([&] { embedGather(ctx, out_1d, table, ids); }));
    check("kernel: gather rejects rows != ids", threw([&] { embedGather(ctx, out_wrong_rows, table, ids); }));
    check("kernel: gather rejects a hidden-size mismatch",
          threw([&] { embedGather(ctx, out, a.allocTensor(DType::BF16, {V, H / 2}), ids); }));

    check("kernel: rmsNorm rejects a non-BF16 out", threw([&] { rmsNorm(ctx, out_f32, out, weight, 1e-6f); }));
    check("kernel: rmsNorm rejects a shape mismatch",
          threw([&] { rmsNorm(ctx, out_wrong_rows, out, weight, 1e-6f); }));
    check("kernel: rmsNorm rejects a wrong-length weight",
          threw([&] { rmsNorm(ctx, norm_out, out, a.allocTensor(DType::BF16, {H / 2}), 1e-6f); }));
    check("kernel: rmsNorm rejects a 2-D weight",
          threw([&] { rmsNorm(ctx, norm_out, out, a.allocTensor(DType::BF16, {2, H}), 1e-6f); }));

    // 3. poison the output, so a kernel that writes nothing cannot pass by
    //    inheriting whatever was in the arena. 0xFF bytes read back as NaN.
    CUDA_CHECK(cudaMemset(out.data, 0xFF, out.nbytes()));
    embedGather(ctx, out, table, ids);

    // Bit-exact: the gather is a 2-byte copy, so any difference is an indexing bug.
    compare("kernel: gather matches the host gather, bit-exact", ctx, out, want_view, 0.0);

    // Negative control: ignoring ids (row t instead of ids[t]) is the mistake this
    // check must be able to see. ids = {5,0,7,2}, so rows 0..3 are the wrong rows.
    std::vector<uint16_t> identity_want(static_cast<size_t>(T * H));
    for (int64_t t = 0; t < T; ++t)
        std::memcpy(&identity_want[static_cast<size_t>(t * H)], &table_host[static_cast<size_t>(t * H)],
                    static_cast<size_t>(H) * 2);
    expect_differs("kernel: control — gathering row t instead of ids[t] differs", ctx, out,
                   hostView(identity_want.data(), DType::BF16, {T, H}), 0.0);
}

// Bad input must throw, not crash or read out of bounds.
void checkBadFile() {
    bool threw = false;
    try { SafeTensors st(dumpPath(L"does_not_exist.safetensors")); } catch (const std::exception&) { threw = true; }
    check("missing file throws", threw);
}

// testing embedGather kernel with real model weights
void checkEmbedGather() {
    SafeTensors shard(modelPath(MODEL_NAME + L"/model.safetensors-00001-of-00001.safetensors"));
    shard.loadTensorsToViews();
    const TensorView& embed_table = shard.at("model.language_model.embed_tokens.weight");
    // Tensor below takes its shape from this view, so a renamed or reshaped key
    // would have the kernel computing row * hidden with the wrong hidden and
    // reading the wrong memory, silently. Assert it as a check, not a warning.
    check("embed: table is BF16 [248320, 2048]", hasShape(embed_table, {248320, 2048}, DType::BF16));

    SafeTensors dump(dumpPath(L"oracle_bf16.safetensors"));
    dump.loadTensorsToViews();
    const TensorView& ids_view = dump.at("ids");
    const int64_t T = ids_view.shape[0];
    const int64_t vocab = embed_table.shape[0];

    // Bounds-check the ids on the HOST. On the device a bad id is a silent read
    // somewhere inside (or past) a 1 GiB buffer -- no exception, no message.
    std::vector<int64_t> ids_host(static_cast<size_t>(T));
    std::memcpy(ids_host.data(), ids_view.host_ptr, ids_view.nbytes);
    bool ids_in_range = true;
    for (int64_t id : ids_host) ids_in_range &= (id >= 0 && id < vocab);
    check("embed: every dumped id indexes inside the table", ids_in_range);

    Arena a(16 << 20);
    Context ctx(a);

    const size_t frees_before = DeviceBuffer::getFreeCount();
    {
        // The table is 1017 MiB and lives as long as the engine does, so it gets
        // its own buffer -- the arena is for per-step activations and is reset.
        DeviceBuffer table_buffer(embed_table.nbytes);
        table_buffer.upload(embed_table.host_ptr, embed_table.nbytes);
        Tensor table(table_buffer.getDevicePtr(), embed_table.dtype, embed_table.shape, embed_table.ndim);

        Tensor ids = uploadBytes(a, ids_host.data(), DType::I64, {T});
        Tensor out = a.allocTensor(DType::BF16, {T, embed_table.shape[1]});

        // 0xFF reads back as NaN, so a kernel that writes nothing (or only part of
        // the output) cannot pass by inheriting what the arena already held.
        CUDA_CHECK(cudaMemset(out.data, 0xFF, out.nbytes()));
        embedGather(ctx, out, table, ids);
        compare("embed: matches the dump, bit-exact", ctx, out, dump.at("embed"), 0.0);

        // Control: rotate the ids by one. Same table, same kernel, same shapes --
        // only *which* row each token asks for changes. This is what proves the
        // check is sensitive to ids on the real 248320-row table, not just on the
        // 8-row synthetic one.
        std::vector<int64_t> rotated(ids_host.begin() + 1, ids_host.end());
        rotated.push_back(ids_host.front());
        Tensor ids_rot = uploadBytes(a, rotated.data(), DType::I64, {T});
        Tensor out_rot = a.allocTensor(DType::BF16, {T, embed_table.shape[1]});
        CUDA_CHECK(cudaMemset(out_rot.data, 0xFF, out_rot.nbytes()));
        embedGather(ctx, out_rot, table, ids_rot);
        expect_differs("embed: control — rotated ids do not match the dump", ctx, out_rot,
                       dump.at("embed"), 0.0);
    }  // table_buffer's destructor runs HERE: the 1017 MiB is back before anything else

    check("embed: the table buffer was freed", DeviceBuffer::getFreeCount() == frees_before + 1);
}

void checkRMSNorm() {
    SafeTensors shard(modelPath(MODEL_NAME + L"/model.safetensors-00001-of-00001.safetensors"));
    shard.loadTensorsToViews();

    const TensorView& norm_weight = shard.at("model.language_model.norm.weight");

    check("rmsnorm: weight is BF16 [2048]", hasShape(norm_weight, {2048}, DType::BF16));

    SafeTensors dump(dumpPath(L"oracle_bf16.safetensors"));
    dump.loadTensorsToViews();

    const TensorView& in_view = dump.at("L23.out");
    const int64_t rows = in_view.shape[0], cols = in_view.shape[1];

    check("rmsnorm: L23.out is BF16 [30, 2048]", hasShape(in_view, {30, 2048}, DType::BF16));

    Arena a(16 << 20);
    Context ctx(a);

    Tensor in = uploadBytes(a, in_view.host_ptr, DType::BF16, {rows, cols});
    Tensor weight = uploadBytes(a, norm_weight.host_ptr, DType::BF16, {cols});
    Tensor norm_out = a.allocTensor(DType::BF16, {rows, cols});

    CUDA_CHECK(cudaMemset(norm_out.data, 0xFF, norm_out.nbytes()));

    rmsNorm(ctx, norm_out, in, weight, 1e-6f);

    compare("rmsnorm: final_norm matches the dump", ctx, norm_out, dump.at("final_norm"), 1e-5);

    Tensor out_plain = a.allocTensor(DType::BF16, {rows, cols});

    CUDA_CHECK(cudaMemset(out_plain.data, 0xFF, out_plain.nbytes()));

    rmsNorm(ctx, out_plain, in, weight, 1e-6f, false);

    expect_differs("rmsnorm: control -- plain w (no 1+) does not match", ctx, out_plain, dump.at("final_norm"), 1e-5);
}

void checkTiedHead() {
    const int64_t T = 2, H = 3, V = 4;

    const float h_f[T * H] = {1,2,3,
                                4, 5, 6};

    const float tab_f[V * H] = {1, 0, 0,
                                0, 1, 0,
                                0, 0, 1,
                                1, 1, 1};

    const float want_f[T * V] = {1, 2, 3,  6,
                                4, 5, 6,  15};

    uint16_t h_b[T * H], tab_b[V * H], want_b[T * V];
    for (int i = 0; i < T * H; ++i) h_b[i]    = f32_to_bf16(h_f[i]);
    for (int i = 0; i < V * H; ++i) tab_b[i]  = f32_to_bf16(tab_f[i]);
    for (int i = 0; i < T * V; ++i) want_b[i] = f32_to_bf16(want_f[i]);

    Arena a(1 << 20);
    Context ctx(a);

    Tensor h = uploadBytes(a, h_b, DType::BF16, {T, H});
    Tensor table = uploadBytes(a, tab_b, DType::BF16, {V, H});
    Tensor logits = a.allocTensor(DType::BF16, {T, V});
    CUDA_CHECK(cudaMemset(logits.data, 0xFF, logits.nbytes()));

    tiedHead(ctx, logits, h, table);
    compare("head/toy: 2x3 @ (4x3)T is bit-exact", ctx, logits, hostView(want_b, DType::BF16, {T, V}), 0.0);

    Tensor bad = a.allocTensor(DType::BF16, {V, T});

    check("head/toy: control -- transposed operands are rejected", threw([&] { tiedHead(ctx, bad, h, table); }));
    check("head/toy: rejects a non-BF16 logits",
        threw([&] { tiedHead(ctx, a.allocTensor(DType::F32, {T, V}), h, table); }));
    check("head/toy: rejects a hidden-size mismatch",
        threw([&] { tiedHead(ctx, logits, h, a.allocTensor(DType::BF16, {V, H + 1})); }));
}


// Task 7 at real scale: the calibration that fixes the bf16 GEMM tolerance every
// later brief inherits. Behind the slow gate -- it uploads the 1017 MiB table.
void checkTiedHeadReal() {
    SafeTensors shard(modelPath(MODEL_NAME + L"/model.safetensors-00001-of-00001.safetensors"));
    shard.loadTensorsToViews();
    const TensorView& table_view = shard.at("model.language_model.embed_tokens.weight");
    check("head: table is BF16 [248320, 2048]", hasShape(table_view, {248320, 2048}, DType::BF16));

    SafeTensors dump(dumpPath(L"oracle_bf16.safetensors"));
    dump.loadTensorsToViews();
    const TensorView& h_view = dump.at("final_norm");
    const TensorView& ref_view = dump.at("logits");
    const int64_t T = h_view.shape[0], H = h_view.shape[1], V = table_view.shape[0];
    check("head: h is BF16 [30, 2048]", hasShape(h_view, {30, 2048}, DType::BF16));
    check("head: logits ref is BF16 [30, 248320]", hasShape(ref_view, {T, V}, DType::BF16));

    Arena a(32 << 20);  // logits alone is T*V*2 = 14.2 MiB
    Context ctx(a);

    const size_t frees_before = DeviceBuffer::getFreeCount();
    {
        // This table is the embedding table AND the tied head weight. 1017 MiB, so
        // it gets its own buffer rather than the per-step arena.
        DeviceBuffer table_buffer(table_view.nbytes);
        table_buffer.upload(table_view.host_ptr, table_view.nbytes);
        Tensor table(table_buffer.getDevicePtr(), table_view.dtype, table_view.shape, table_view.ndim);

        Tensor h = uploadBytes(a, h_view.host_ptr, DType::BF16, {T, H});
        Tensor logits = a.allocTensor(DType::BF16, {T, V});

        // beta == 0 means cuBLAS never reads C, so poisoning is safe here and still
        // catches a GEMM that did not run: every element would stay NaN, and
        // CompareResult::within() refuses to forgive nan_one.
        CUDA_CHECK(cudaMemset(logits.data, 0xFF, logits.nbytes()));

        tiedHead(ctx, logits, h, table);

        // THE bf16 GEMM BUDGET. Measured 2026-10-29: rel = 2.747e-03 (max|d| 0.0625,
        // max|ref| 22.75, worst[168758] -8.75 vs -8.81). Gate set to the next round
        // number above it. Not bit-exact and never will be: cuBLAS chooses its own
        // accumulation order over k=2048. Every later stage inherits this budget, so
        // if this number moves, something changed in the GEMM path -- not in tolerance.
        compare("head: logits vs the dump", ctx, logits, ref_view, 5e-3);

        // argmax is the check a tolerance cannot make, and the only one here that
        // would notice a transposed [V, T] result: compareToRef compares element
        // COUNT, not shape, and both layouts hold the same 7,449,600 elements.
        ctx.sync();
        std::vector<uint8_t> mine(logits.nbytes());
        CUDA_CHECK(cudaMemcpy(mine.data(), logits.data, logits.nbytes(), cudaMemcpyDeviceToHost));

        size_t argmax_mismatch = 0;
        for (int64_t t = 0; t < T; ++t) {
            int64_t best_mine = 0, best_ref = 0;
            double val_mine = -1e300, val_ref = -1e300;
            for (int64_t v = 0; v < V; ++v) {
                const size_t i = static_cast<size_t>(t) * static_cast<size_t>(V) + static_cast<size_t>(v);
                const double m = elemAsDouble(mine.data(), DType::BF16, i);
                const double r = elemAsDouble(ref_view.host_ptr, DType::BF16, i);
                if (m > val_mine) { val_mine = m; best_mine = v; }
                if (r > val_ref) { val_ref = r; best_ref = v; }
            }
            argmax_mismatch += (best_mine != best_ref);
        }
        check("head: argmax matches the dump on all 30 rows", argmax_mismatch == 0);
    }  // table_buffer's destructor runs HERE

    check("head: the table buffer was freed", DeviceBuffer::getFreeCount() == frees_before + 1);
}

}  // namespace

int main() {
    char *deltaforge_slow_tests = nullptr;
    _dupenv_s(&deltaforge_slow_tests, nullptr, "DELTAFORGE_SLOW_TESTS");
    try {
        checkBf16Dump();
        checkFp32Dump();
        checkBadFile();
        checkBf16Helpers();
        checkArena();
        checkDeviceBufferCounters();
        checkArenaReuse();
        checkCompare();
        checkKernelPlumbing();
        checkTiedHead();
        checkRMSNorm();
        if(deltaforge_slow_tests != nullptr && std::string(deltaforge_slow_tests) == "1"){
            std::cout << "Running slow tests" << std::endl;
            checkEmbedGather();
            checkTiedHeadReal();
        }else {
            std::cout << "Skipping slow tests" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "unexpected exception: " << e.what() << '\n';
        return 1;
    }

    std::cout << (failures ? "\nFAILED: " : "\nall checks passed: ") << failures << " failing\n";
    return failures;
}
