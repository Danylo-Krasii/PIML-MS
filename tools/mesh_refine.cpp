// Solve six unit macroscopic strains on a voxel grain map with the in-house
// Moulinec-Suquet solver (fft_homog.hpp) and dump the stress fields.
//
// Inputs are already in the solver's own Mandel component order; the Python
// caller owns every permutation. Binary layout of every array is the solver's
// (z*N + y)*N + x, x fastest.
//
//   mesh_refine N NMAT phase.i32 C.f64 tol maxit out_prefix [lambda0 mu0]
//
// Writes out_prefix.k.f64 (6*N^3 doubles, stress component-major) and
// out_prefix.k.txt (iterations, status, seconds, <sigma>) for k = 0..5.
// With lambda0 and mu0 given (Pa) the reference medium is fixed instead of
// picked by the solver, so that a run with tol 0 and maxit K is the Born
// iterate K for a recorded C0. The fixed point does not depend on C0.
#include "fft_homog.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using ffth::Mat6; using ffth::Vec6;

template <class T> static std::vector<T> read_bin(const std::string& p, std::size_t n) {
    std::vector<T> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T))) {
        std::fprintf(stderr, "cannot read %zu items from %s\n", n, p.c_str()); std::exit(2);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc != 8 && argc != 10) { std::fprintf(stderr, "usage: %s N NMAT phase.i32 C.f64 tol maxit out_prefix [lambda0 mu0]\n", argv[0]); return 1; }
    const int N = std::atoi(argv[1]), NMAT = std::atoi(argv[2]);
    const double tol = std::atof(argv[5]); const int maxit = std::atoi(argv[6]);
    const std::string out = argv[7];
    const std::size_t NV = std::size_t(N) * N * N;
    auto phase = read_bin<int>(argv[3], NV);
    auto cflat = read_bin<double>(argv[4], std::size_t(NMAT) * 36);
    std::vector<Mat6> C(NMAT);
    for (int m = 0; m < NMAT; ++m) for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) C[m][a][b] = cflat[m * 36 + a * 6 + b];

    ffth::FFTHomogenizer H(N, N, N);
    H.set_materials(C); H.set_phase_field(phase);
    H.set_tolerance(tol); H.set_max_iterations(maxit);
    if (argc == 10) H.set_reference(std::atof(argv[8]), std::atof(argv[9]));
    for (int k = 0; k < 6; ++k) {
        Vec6 E{}; E[k] = 1e-4;
        int it = 0;
        auto t0 = std::chrono::steady_clock::now();
        Vec6 avg = H.solve(E, &it);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<double> field(6 * NV);
        for (int c = 0; c < 6; ++c) { const auto& s = H.stress_component(c); for (std::size_t i = 0; i < NV; ++i) field[c * NV + i] = s[i].real(); }
        std::ofstream(out + "." + std::to_string(k) + ".f64", std::ios::binary).write(reinterpret_cast<const char*>(field.data()), field.size() * sizeof(double));
        std::ofstream txt(out + "." + std::to_string(k) + ".txt");
        txt << it << " " << (it >= maxit ? "NOTCONVERGED" : "converged") << " " << secs << "\n";
        for (int c = 0; c < 6; ++c) txt << avg[c] << (c < 5 ? " " : "\n");
        std::printf("N=%d load %d: %d iters, %.1f s%s\n", N, k, it, secs, it >= maxit ? "  NOT CONVERGED" : "");
        std::fflush(stdout);
    }
    return 0;
}
