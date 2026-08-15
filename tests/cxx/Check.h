/**
 * \file Check.h
 * \brief A three-macro assertion helper for the libKalman test binaries.
 *
 * The Python suite runs under pytest; these C++ tests are plain executables
 * registered with CTest, so they carry no framework dependency at all. Each one
 * returns the number of failed checks, and CTest reports non-zero as a failure.
 */
#ifndef KALMAN_TESTS_CHECK_H
#define KALMAN_TESTS_CHECK_H

#include <cmath>
#include <cstdio>

namespace Check
{
    inline int &Failures()
    {
        static int failures = 0;
        return failures;
    }

    inline void Report(bool ok, const char *what)
    {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++Failures();
    }

    inline void ReportClose(double got, double want, double tol, const char *what)
    {
        const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
        if (ok) std::printf("  %-62s ok\n", what);
        else
        {
            std::printf("  %-62s FAIL (got %.12g, want %.12g +/- %g)\n", what, got, want, tol);
            ++Failures();
        }
    }

    inline int Summary(const char *name)
    {
        const int n = Failures();
        std::printf("%s: %s (%d failure%s)\n\n", name, n ? "FAILED" : "passed", n, n == 1 ? "" : "s");
        return n ? 1 : 0;
    }
}

#define CHECK(cond, what)                  Check::Report((cond), (what))
#define CHECK_CLOSE(got, want, tol, what)  Check::ReportClose((got), (want), (tol), (what))

#endif // KALMAN_TESTS_CHECK_H
