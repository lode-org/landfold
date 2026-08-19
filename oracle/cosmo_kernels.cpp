/* SPDX-License-Identifier: GPL-3.0-only
 * Verbatim kernels from lab-cosmo/sketchmap tools/libdimred.cpp
 * (Ceriotti 2011). Test oracle only; not linked into landfold. */
#include <cmath>
#include <cstdio>
#include <valarray>
#include <vector>

enum Mode { Identity, Sigmoid, Compress, XSigmoid, Warp };

struct Fn {
    Mode mode;
    std::valarray<double> pars;
    void set(Mode m, const std::valarray<double>& npars) {
        mode = m;
        switch (m) {
        case Identity:
            pars.resize(0);
            break;
        case Compress:
            pars.resize(1);
            pars[0] = 1.0 / npars[0];
            break;
        case Sigmoid:
            pars.resize(2);
            pars[0] = 1.0 / npars[0];
            pars[1] = 2.0 * pars[0] * pars[0];
            break;
        case XSigmoid:
            pars.resize(5);
            pars[0] = 1.0 / npars[0];
            pars[1] = std::pow(2., npars[1] / npars[2]) - 1.0;
            pars[2] = npars[1];
            pars[3] = npars[2];
            pars[4] = -npars[2] / npars[1];
            break;
        case Warp:
            pars.resize(10);
            pars[0] = 1.0 / npars[0];
            pars[1] = std::pow(2., npars[1] / npars[2]) - 1.0;
            pars[2] = npars[1];
            pars[3] = npars[2];
            pars[4] = -npars[2] / npars[1];
            pars[5] = npars[0];
            pars[6] = std::pow(2., npars[3] / npars[4]) - 1.0;
            pars[7] = 1.0 / npars[3];
            pars[8] = npars[4];
            pars[9] = -npars[3] / npars[4];
            break;
        }
    }
    void xsig(double x, double& rf, double& rdf) const {
        if (x == 0.0) {
            rf = 0.0;
            rdf = 0.0;
            return;
        }
        double sx = x * pars[0];
        sx = pars[1] * std::pow(sx, pars[2]);
        rf = std::pow(1.0 + sx, pars[4]);
        rdf = pars[3] * sx / x * rf / (1.0 + sx);
        rf = 1.0 - rf;
    }
    void fdf(double x, double& rf, double& rdf) const {
        switch (mode) {
        case Identity:
            rf = x;
            rdf = 1.0;
            break;
        case Compress: {
            double sx = x * pars[0];
            sx = 1.0 / (1.0 + sx);
            rf = 1.0 - sx;
            rdf = (sx * sx) * pars[0];
            break;
        }
        case Sigmoid: {
            double sx = x * pars[0];
            sx = 1.0 / (1.0 + sx * sx);
            rf = 1.0 - sx;
            rdf = x * (sx * sx) * pars[1];
            break;
        }
        case XSigmoid:
            xsig(x, rf, rdf);
            break;
        case Warp: {
            double fx, dfx;
            xsig(x, fx, dfx);
            double sx = std::pow(1.0 - fx, pars[9]);
            sx = (sx - 1.0) / pars[6];
            rf = pars[5] * std::pow(sx, pars[7]);
            double den = std::pow(1.0 - fx, -pars[9]);
            den = (den - 1.0) * (fx - 1.0) * pars[8];
            rdf = rf / den * dfx;
            break;
        }
        }
    }
};

static void dump_xfer(const char* tag, Fn& fn) {
    const double xs[] = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 6.0, 8.0, 10.0, 20.0};
    std::printf("BEGIN xfer %s\n", tag);
    for (double x : xs) {
        double v = 0, d = 0;
        fn.fdf(x, v, d);
        std::printf("%.17e %.17e %.17e\n", x, v, d);
    }
    std::printf("END\n");
}

static double euclid(const double* a, const double* b, unsigned n) {
    double d = 0.0;
    for (unsigned i = 0; i < n; ++i) d += (b[i] - a[i]) * (b[i] - a[i]);
    return std::sqrt(d);
}

static double pbc(const double* a, const double* b, const double* per, unsigned n) {
    double d = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        double dx = (b[i] - a[i]) / per[i];
        dx -= std::round(dx);
        dx *= per[i];
        d += dx * dx;
    }
    return std::sqrt(d);
}

static double dotd(const double* a, const double* b, unsigned n) {
    double d = 0.0;
    for (unsigned i = 0; i < n; ++i) d += b[i] * a[i];
    return -std::log(d);
}

int main() {
    Fn fn;
    std::valarray<double> none;
    fn.set(Identity, none);
    dump_xfer("identity", fn);
    std::valarray<double> s1(1);
    s1[0] = 1.0;
    fn.set(Sigmoid, s1);
    dump_xfer("sigmoid_1", fn);
    fn.set(Compress, s1);
    dump_xfer("compress_1", fn);
    std::valarray<double> xs(3);
    xs[0] = 5;
    xs[1] = 8;
    xs[2] = 1;
    fn.set(XSigmoid, xs);
    dump_xfer("xsigmoid_5_8_1", fn);
    xs[0] = 6;
    xs[1] = 8;
    xs[2] = 8;
    fn.set(XSigmoid, xs);
    dump_xfer("xsigmoid_6_8_8", fn);
    std::valarray<double> wp(5);
    wp[0] = 5;
    wp[1] = 8;
    wp[2] = 1;
    wp[3] = 2;
    wp[4] = 2;
    fn.set(Warp, wp);
    dump_xfer("warp_5_8_1_2_2", fn);

    const double a3[] = {0, 0, 0};
    const double b3[] = {3, 4, 0};
    std::printf("BEGIN metric euclid_3_4_5\n%.17e\nEND\n", euclid(a3, b3, 3));
    const double pa[] = {0.05};
    const double pb[] = {0.95};
    const double per[] = {1.0};
    std::printf("BEGIN metric pbc_wrap\n%.17e\nEND\n", pbc(pa, pb, per, 1));
    const double u[] = {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    std::printf("BEGIN metric dot_self\n%.17e\nEND\n", dotd(u, u, 2));
    return 0;
}
