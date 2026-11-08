#include "kernel.h"
#include <VX_config.h>
#include <VX_types.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(vx_result_t result, const char* operation) {
    if (result != VX_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + vx_result_string(result));
}
#define CHECK(expr) check((expr), #expr)

struct Counter { const char* name; uint32_t group, address; };
#define CORE(name) {"core." #name, VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_##name}
#define DCACHE(name) {"dcache." #name, VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_##name}
static constexpr Counter counters[] = {
    {"cycles", VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE},
    {"warp_instructions", VX_DCR_MPM_CLASS_BASE, VX_CSR_MINSTRET},
    CORE(SCHED_IDLE), CORE(ACTIVE_WARPS), CORE(STALLED_WARPS),
    CORE(ISSUED_WARPS), CORE(ISSUED_THREADS),
    CORE(STALL_FETCH), CORE(STALL_IBUF), CORE(STALL_SCRB), CORE(STALL_OPDS),
    CORE(STALL_ALU), CORE(STALL_FPU), CORE(STALL_LSU), CORE(STALL_SFU), CORE(STALL_TCU),
    CORE(BRANCHES), CORE(DIVERGENCE),
    CORE(INSTR_ALU), CORE(INSTR_FPU), CORE(INSTR_LSU), CORE(INSTR_SFU), CORE(INSTR_TCU),
    CORE(IFETCHES), CORE(IFETCH_LT), CORE(LOADS), CORE(LOAD_LT), CORE(STORES),
    {"icache.READS", VX_DCR_MPM_CLASS_ICACHE, VX_CSR_MPM_ICACHE_READS},
    {"icache.MISS_R", VX_DCR_MPM_CLASS_ICACHE, VX_CSR_MPM_ICACHE_MISS_R},
    {"icache.MSHR_ST", VX_DCR_MPM_CLASS_ICACHE, VX_CSR_MPM_ICACHE_MSHR_ST},
    DCACHE(READS), DCACHE(WRITES), DCACHE(MISS_R), DCACHE(MISS_W), DCACHE(EVICTS),
    DCACHE(BANK_ST), DCACHE(MSHR_ST),
};
using Snapshot = std::array<uint64_t, sizeof(counters) / sizeof(counters[0])>;

static Snapshot snapshot(vx_device_h device) {
    Snapshot values{};
    for (size_t i = 0; i < values.size(); ++i)
        CHECK(vx_device_mpm_query(device, counters[i].group, counters[i].address, 0, &values[i]));
    return values;
}

struct Resources {
    vx_device_h device = nullptr;
    vx_queue_h queue = nullptr;
    std::array<vx_buffer_h, 3> buffers{};
    ~Resources() {
        if (queue) vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
        unload_add_kernel();
        for (auto buffer : buffers) if (buffer) vx_buffer_release(buffer);
        if (queue) vx_queue_release(queue);
        if (device) vx_device_release(device);
    }
};

static unsigned argument(const char* text, unsigned limit, const char* name) {
    char* end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (!*text || *end || !value || value > limit)
        throw std::runtime_error(std::string(name) + " must be in [1," + std::to_string(limit) + "]");
    return static_cast<unsigned>(value);
}

int main(int argc, char** argv) try {
    if (argc > 4)
        throw std::runtime_error("usage: run [n=1024] [repeats=3] [output-prefix=add-bench]");
    const unsigned n = argc > 1 ? argument(argv[1], 1000000, "n") : 1024;
    const unsigned repeats = argc > 2 ? argument(argv[2], 100, "repeats") : 3;
    const std::string prefix = argc > 3 ? argv[3] : "add-bench";
    const unsigned line = VX_CFG_DCACHE_LINE_SIZE;
    const unsigned bytes = ((n * sizeof(uint32_t) + line + line - 1) / line) * line;
    const unsigned capacity = bytes / sizeof(uint32_t);
    if (3 * bytes + line > VX_CFG_DCACHE_SIZE)
        throw std::runtime_error("three arrays (with guard lines) plus arguments exceed configured L1 D-cache capacity");
    if (!VX_CFG_DCACHE_ENABLED)
        throw std::runtime_error("benchmark requires L1 D-cache enabled");

    Resources r;
    CHECK(vx_device_open(0, &r.device));
    uint64_t cores, threads, warps;
    CHECK(vx_device_query(r.device, VX_CAPS_NUM_CORES, &cores));
    CHECK(vx_device_query(r.device, VX_CAPS_NUM_THREADS, &threads));
    CHECK(vx_device_query(r.device, VX_CAPS_NUM_WARPS, &warps));
    if (cores != 1 || threads != VX_CFG_NUM_THREADS || warps != VX_CFG_NUM_WARPS)
        throw std::runtime_error("runtime hardware differs from benchmark build configuration");
    vx_queue_info_t info{};
    info.struct_size = sizeof(info);
    CHECK(vx_queue_create(r.device, &info, &r.queue));
    std::array<uint64_t, 3> addresses{};
    for (unsigned i = 0; i < r.buffers.size(); ++i) {
        CHECK(vx_buffer_create(r.device, bytes, VX_MEM_READ_WRITE, &r.buffers[i]));
        CHECK(vx_buffer_address(r.buffers[i], &addresses[i]));
    }
    std::printf("CONFIG driver=%s cores=%" PRIu64 " hw_warps=%" PRIu64 " threads=%" PRIu64
                " simd=%u program_warps=%u block=%u n=%u grid=%u\n",
                std::getenv("VORTEX_DRIVER") ? std::getenv("VORTEX_DRIVER") : "default",
                cores, warps, threads, VX_CFG_SIMD_WIDTH, BENCH_WARPS, BENCH_BLOCK, n, (n + BENCH_BLOCK - 1) / BENCH_BLOCK);
    std::printf("CACHE L1D=%u ways=%u line=%u data=%u allocated_arrays=%u args=32 bytes\n",
                VX_CFG_DCACHE_SIZE, VX_CFG_DCACHE_NUM_WAYS, line, 12 * n, 3 * bytes);
    std::puts("TIMING core0 MCYCLE deltas; RTL counts scheduler-busy cycles, SimX counts core ticks. Includes startup and memory stalls; excludes host transfers.");
    std::puts("CACHE_STATE cold at each launch: runtime flushes between launches. Capacity fit does not imply L1 hits; spills also consume cache.");
    std::puts("COUNTER_SCOPE before/after queue completion; idle/stall counters may include flush/readback clocks. Values are raw, not percentages.");

    std::vector<uint32_t> a(capacity), b(capacity), dst(capacity);
    for (unsigned i = 0; i < capacity; ++i) {
        a[i] = i * 0x1020305u + 0x89abcdefu;
        b[i] = (i * 17u) ^ 0x7fff0000u;
    }
    auto prepare = [&] {
        std::fill(dst.begin(), dst.end(), 0xdeadbeefu);
        CHECK(vx_enqueue_write(r.queue, r.buffers[0], 0, a.data(), bytes, 0, nullptr, nullptr));
        CHECK(vx_enqueue_write(r.queue, r.buffers[1], 0, b.data(), bytes, 0, nullptr, nullptr));
        CHECK(vx_enqueue_write(r.queue, r.buffers[2], 0, dst.data(), bytes, 0, nullptr, nullptr));
        CHECK(vx_queue_finish(r.queue, VX_TIMEOUT_INFINITE));
    };
    auto launch = [&](unsigned count) {
        CHECK(add_kernel(r.device, r.queue, addresses[0], addresses[1], addresses[2], static_cast<int32_t>(count)));
        CHECK(vx_queue_finish(r.queue, VX_TIMEOUT_INFINITE));
    };
    auto verify = [&](unsigned count) {
        CHECK(vx_enqueue_read(r.queue, dst.data(), r.buffers[2], 0, bytes, 0, nullptr, nullptr));
        CHECK(vx_queue_finish(r.queue, VX_TIMEOUT_INFINITE));
        for (unsigned i = 0; i < capacity; ++i) {
            uint32_t expected = i < count ? a[i] + b[i] : 0xdeadbeefu;
            if (dst[i] != expected)
                throw std::runtime_error("wrong result or modified tail at n=" + std::to_string(count) + " index=" + std::to_string(i));
        }
    };

    // Match the Python example's unmeasured launch before collecting deltas.
    prepare();
    CHECK(load_add_kernel(r.device));
    launch(n);

    std::ofstream summary(prefix + ".csv"), raw(prefix + ".counters.csv");
    if (!summary || !raw) throw std::runtime_error("cannot open benchmark CSV output");
    summary << "repeat,n,block,num_warps,hw_warps,grid,cycles,warp_instructions,cycles_per_element,host_launch_ms\n";
    raw << "repeat,counter,delta\n";
    uint64_t min_cycles = UINT64_MAX, max_cycles = 0, total_cycles = 0;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        prepare();
        const auto before = snapshot(r.device);
        const auto start = std::chrono::steady_clock::now();
        launch(n);
        const double host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const auto after = snapshot(r.device);
        verify(n);
        const uint64_t cycles = after[0] - before[0], instructions = after[1] - before[1];
        if (!cycles || !instructions) throw std::runtime_error("device returned empty cycle/instruction counters");
        min_cycles = std::min(min_cycles, cycles);
        max_cycles = std::max(max_cycles, cycles);
        total_cycles += cycles;
        summary << repeat << ',' << n << ',' << BENCH_BLOCK << ',' << BENCH_WARPS << ',' << warps
                << ',' << (n + BENCH_BLOCK - 1) / BENCH_BLOCK
                << ',' << cycles << ',' << instructions << ',' << static_cast<double>(cycles) / n << ',' << host_ms << '\n';
        std::printf("BENCH repeat=%u cycles=%" PRIu64 " warp_instructions=%" PRIu64
                    " cycles/element=%.4f warp_IPC=%.4f PASS\n", repeat, cycles, instructions,
                    static_cast<double>(cycles) / n, static_cast<double>(instructions) / cycles);
        std::printf("HOST repeat=%u launch_and_finish_ms=%.3f (includes argument staging and cache flush)\n", repeat, host_ms);
        for (size_t i = 0; i < after.size(); ++i) {
            uint64_t delta = after[i] - before[i];
            raw << repeat << ',' << counters[i].name << ',' << delta << '\n';
            std::printf("COUNTER repeat=%u %s=%" PRIu64 "\n", repeat, counters[i].name, delta);
        }
    }
    std::printf("RESULT n=%u repeats=%u cycles_min=%" PRIu64 " cycles_mean=%.2f cycles_max=%" PRIu64 "\n",
                n, repeats, min_cycles, static_cast<double>(total_cycles) / repeats, max_cycles);
    std::ofstream output(prefix + ".output.bin", std::ios::binary);
    output.write(reinterpret_cast<const char*>(dst.data()), bytes);
    if (!output) throw std::runtime_error("cannot write output buffer");
    std::printf("FILES %s.csv %s.counters.csv %s.output.bin\n", prefix.c_str(), prefix.c_str(), prefix.c_str());
    std::puts("RUNTIME_REPORT cumulative including warm-up; VORTEX_PROFILING selects report class:");
    CHECK(vx_device_dump_perf(r.device, stdout));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
