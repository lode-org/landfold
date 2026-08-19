/* Dump goldens from NLDRFunction / NLDRMetric / NLDRMDS / NLDRITERChi
 * (interpol off). MDS block values are i<j Euclidean distances of the
 * private embedding p (invariant to eigenvector sign). */
#include "dimreduce.hpp"
#include <cmath>
#include <cstdio>
#include <valarray>

using namespace toolbox;

static void dump_xfer(const char* tag, NLDRFunction& fn) {
    const double xs[] = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 6.0, 8.0, 10.0, 20.0};
    std::printf("BEGIN xfer %s\n", tag);
    for (double x : xs) {
        double v = 0.0, d = 0.0;
        fn.fdf(x, v, d);
        std::printf("%.17e %.17e %.17e\n", x, v, d);
    }
    std::printf("END\n");
}

/* Upper-triangle i<j pairwise distances under a metric. */
static void dump_pair(const char* tag, NLDRMetric& met, const double* data,
                      unsigned long n, unsigned long D) {
    std::printf("BEGIN pair %s\n", tag);
    for (unsigned long i = 0; i < n; ++i) {
        for (unsigned long j = i + 1; j < n; ++j) {
            const double* a = data + i * D;
            const double* b = data + j * D;
            std::printf("%.17e\n", met.dist(a, b, D));
        }
    }
    std::printf("END\n");
}

/* Torgerson MDS: p is private, so dump i<j Euclidean distances of p.
 * Those distances are invariant to eigenvector sign flips. */
static void dump_mds(const char* tag, const double* data, unsigned long n,
                     unsigned long D, unsigned long lowdim) {
    FMatrix<double> pts(n, D, 0.0);
    for (unsigned long i = 0; i < n; ++i)
        for (unsigned long h = 0; h < D; ++h) pts(i, h) = data[i * D + h];

    NLDRMetricEuclid eu;
    NLDRMDSOptions opts;
    opts.metric = &eu;
    opts.mode = MDS;
    opts.lowdim = lowdim;
    opts.verbose = false;
    NLDRProjection proj;
    NLDRMDSReport report;
    NLDRMDS(pts, proj, opts, report);

    std::valarray<std::valarray<double> > nP, np;
    proj.get_points(nP, np);

    std::printf("BEGIN mds %s\n", tag);
    for (unsigned long i = 0; i < n; ++i) {
        for (unsigned long j = i + 1; j < n; ++j) {
            std::printf("%.17e\n", eu.dist(&np[i][0], &np[j][0], lowdim));
        }
    }
    std::printf("END\n");
}

/* Three-point χ / ∇χ from the real NLDRITERChi object. */
static void dump_chi(const char* tag, double imix, NLDRFunction& tfun) {
    const unsigned long n = 3, d = 2;
    FMatrix<double> hd(n, n, 0.0), fhd(n, n, 0.0);
    const double raw[3] = {1.0, 2.0, 1.5}; /* (0,1), (0,2), (1,2) */
    const unsigned long ia[3] = {0, 0, 1};
    const unsigned long ib[3] = {1, 2, 2};
    for (unsigned long k = 0; k < 3; ++k) {
        double fv = 0.0, df = 0.0;
        tfun.fdf(raw[k], fv, df);
        hd(ia[k], ib[k]) = hd(ib[k], ia[k]) = raw[k];
        fhd(ia[k], ib[k]) = fhd(ib[k], ia[k]) = fv;
    }

    NLDRMetricEuclid eu;
    NLDRITERChi chi;
    chi.n = n;
    chi.d = d;
    chi.imix = imix;
    chi.dogradient = true;
    chi.metric = &eu;
    chi.tfun = tfun;
    chi.set_hd(hd, fhd);

    std::valarray<double> coords(n * d);
    const double packed[6] = {0.0, 0.0, 0.8, 0.1, -0.2, 0.7};
    for (unsigned long i = 0; i < n * d; ++i) coords[i] = packed[i];
    chi.set_vars(coords);

    double val = 0.0;
    std::valarray<double> grad;
    chi.get_value(val);
    chi.get_gradient(grad);

    std::printf("BEGIN chi %s\n", tag);
    std::printf("%.17e\n", val);
    for (unsigned long i = 0; i < grad.size(); ++i) std::printf("%.17e\n", grad[i]);
    std::printf("END\n");
}

int main() {
    std::valarray<double> none;
    NLDRFunction id;
    id.set_mode(NLDRIdentity, none, false);
    dump_xfer("identity", id);

    std::valarray<double> s1(1);
    s1[0] = 1.0;
    NLDRFunction sig;
    sig.set_mode(NLDRSigmoid, s1, false);
    dump_xfer("sigmoid_1", sig);

    NLDRFunction cmp;
    cmp.set_mode(NLDRCompress, s1, false);
    dump_xfer("compress_1", cmp);

    std::valarray<double> xs(3);
    xs[0] = 5.0;
    xs[1] = 8.0;
    xs[2] = 1.0;
    NLDRFunction xsig;
    xsig.set_mode(NLDRXSigmoid, xs, false);
    dump_xfer("xsigmoid_5_8_1", xsig);

    xs[0] = 6.0;
    xs[1] = 8.0;
    xs[2] = 8.0;
    NLDRFunction xsig2;
    xsig2.set_mode(NLDRXSigmoid, xs, false);
    dump_xfer("xsigmoid_6_8_8", xsig2);

    std::valarray<double> wp(5);
    wp[0] = 5.0;
    wp[1] = 8.0;
    wp[2] = 1.0;
    wp[3] = 2.0;
    wp[4] = 2.0;
    NLDRFunction wrp;
    wrp.set_mode(NLDRWarp, wp, false);
    dump_xfer("warp_5_8_1_2_2", wrp);

    NLDRMetricEuclid eu;
    const double a3[] = {0.0, 0.0, 0.0};
    const double b3[] = {3.0, 4.0, 0.0};
    std::printf("BEGIN metric euclid_3_4_5\n%.17e\nEND\n", eu.dist(a3, b3, 3));

    NLDRMetricPBC pbc;
    pbc.periods.resize(1);
    pbc.periods = 1.0;
    const double pa[] = {0.05};
    const double pb[] = {0.95};
    std::printf("BEGIN metric pbc_wrap\n%.17e\nEND\n", pbc.dist(pa, pb, 1));

    const double tri[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    dump_pair("euclid_triangle", eu, tri, 3, 2);
    dump_mds("torgerson_triangle", tri, 3, 2, 2);

    const double tetra[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    dump_pair("euclid_tetra", eu, tetra, 4, 3);
    dump_mds("torgerson_tetra", tetra, 4, 3, 2);

    const double pbc3[] = {0.05, 0.50, 0.95};
    dump_pair("pbc_3pt", pbc, pbc3, 3, 1);

    const double u[] = {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    NLDRMetricDot dot;
    std::printf("BEGIN metric dot_self\n%.17e\nEND\n", dot.dist(u, u, 2));

    std::valarray<double> xs14(3);
    xs14[0] = 1.0;
    xs14[1] = 4.0;
    xs14[2] = 3.0;
    NLDRFunction xsig14;
    xsig14.set_mode(NLDRXSigmoid, xs14, false);
    dump_chi("xsig_1_4_3_imix01", 0.1, xsig14);
    dump_chi("xsig_1_4_3_imix00", 0.0, xsig14);

    NLDRFunction idchi;
    std::valarray<double> none2;
    idchi.set_mode(NLDRIdentity, none2, false);
    dump_chi("identity_imix00", 0.0, idchi);
    return 0;
}
