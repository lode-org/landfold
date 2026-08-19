/* Same LCG point set as examples/bench_chi.rs. Times NLDRITERChi::set_vars. */
#include "dimreduce.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <valarray>

using namespace toolbox;

static double lcg(std::uint64_t& state) {
    state = state * 6364136223846793005ull + 1ull;
    return double(state >> 11) * (1.0 / double(1ull << 53));
}

int main(int argc, char** argv) {
    const unsigned long n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 400;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 40;
    const unsigned long dim_hi = 8, d = 2;
    std::uint64_t st = 1;
    FMatrix<double> hi(n, dim_hi, 0.0);
    for (unsigned long i = 0; i < n; ++i)
        for (unsigned long h = 0; h < dim_hi; ++h) hi(i, h) = lcg(st);

    NLDRMetricEuclid eu;
    FMatrix<double> hd(n, n, 0.0), fhd(n, n, 0.0);
    std::valarray<double> xs(3);
    xs[0] = 5.0;
    xs[1] = 8.0;
    xs[2] = 1.0;
    NLDRFunction thd;
    thd.set_mode(NLDRXSigmoid, xs, false);
    for (unsigned long i = 0; i < n; ++i) {
        hd(i, i) = 0.0;
        fhd(i, i) = 0.0;
        for (unsigned long j = 0; j < i; ++j) {
            double r = eu.dist(&hi(i, 0), &hi(j, 0), dim_hi);
            double fv = 0.0, df = 0.0;
            thd.fdf(r, fv, df);
            hd(i, j) = hd(j, i) = r;
            fhd(i, j) = fhd(j, i) = fv;
        }
    }
    xs[1] = 2.0;
    xs[2] = 2.0;
    NLDRFunction tld;
    tld.set_mode(NLDRXSigmoid, xs, false);

    NLDRITERChi chi;
    chi.n = n;
    chi.d = d;
    chi.imix = 0.0;
    chi.dogradient = true;
    chi.metric = &eu;
    chi.tfun = tld;
    chi.set_hd(hd, fhd);

    std::valarray<double> coords(n * d);
    for (unsigned long i = 0; i < n * d; ++i) coords[i] = lcg(st) - 0.5;
    chi.set_vars(coords);
    double val = 0.0;
    chi.get_value(val);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) chi.set_vars(coords);
    auto t1 = std::chrono::steady_clock::now();
    chi.get_value(val);
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / reps;
    std::printf("impl=cpp n=%lu reps=%d ns_per_eval=%lld chi=%.17e\n",
                n, reps, static_cast<long long>(ns), val);
    return 0;
}
