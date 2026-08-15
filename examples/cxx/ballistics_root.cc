/**
 * \file ballistics_root.cc
 * \brief libKalmanFramework driven from a ROOT program — the integration demonstrator.
 *
 * A projectile under gravity and quadratic drag, tracked by an extended Kalman
 * filter, with the result drawn through ROOT. It exists to show three things at
 * once:
 *
 *  1. **The dependency direction.** This translation unit includes both
 *     `<Kalman/Kalman.h>` and `<TGraph.h>`. libKalmanFramework itself includes
 *     neither ROOT nor anything else — ROOT links *it*, never the reverse. That
 *     is what lets the same library also serve a non-ROOT analysis.
 *  2. **Where physics lives.** `Ballistics` below is an *application*: it
 *     subclasses Kalman::Models::Extended and encodes gravity, drag and a
 *     launch. The framework ships no such model, exactly as `modules/framework/`
 *     ships none and `modules/apps/ballistics/` does.
 *  3. **Where plotting lives.** Kalman::Record hands out plain arrays; this file
 *     turns them into TGraphs. The library never grew a drawing dependency to
 *     make that possible.
 *
 *     source var/install/bin/thislib.KalmanFramework.sh   # or set the paths yourself
 *     ./var/build/bin/ballistics_root
 */
#include <Kalman/Kalman.h>

#include <TAxis.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TStyle.h>

#include <cmath>
#include <cstdio>
#include <vector>

/**
 * \brief Projectile with gravity and quadratic drag — an application model.
 *
 * State is axis-major `[px, py, pz, vx, vy, vz]`; only position is measured.
 * The dynamics are genuinely nonlinear: drag goes as `|v| v`, so the Jacobian
 * depends on the state and the extended engine has to linearise every step.
 */
class Ballistics : public Kalman::Models::Extended
{
    private:
        double fGravity = 9.81;    // g [m/s^2]
        double fDragOverMass = 0.; // c_drag / m [1/m]

    public:
        Ballistics(double fs, double dragOverMass)
            : Kalman::Models::Extended("ballistics", 6, 3), fDragOverMass(dragOverMass)
        {
            // Models::Extended already declared the extended engine and switched
            // the Jacobian to central differences, which is what a nonlinear f()
            // needs -- F(t) is not its Jacobian.
            SetSampleRate(fs);

            Kalman::Matrix H(3, 6, 0.);
            for (std::size_t i = 0; i < 3; ++i) H(i, i) = 1.;   // observe position
            SetH(H);

            const double dt = DeltaT();
            const double q = 0.5 * 0.5;                          // sigma_a^2 [m/s^2]^2

            // White-noise acceleration, per axis, coupling position and velocity.
            Kalman::Matrix Q(6, 6, 0.);
            for (std::size_t i = 0; i < 3; ++i)
            {
                Q(i, i)         = q * dt * dt * dt * dt / 4.;
                Q(i, i + 3)     = q * dt * dt * dt / 2.;
                Q(i + 3, i)     = Q(i, i + 3);
                Q(i + 3, i + 3) = q * dt * dt;
            }
            SetQ(Q);

            Kalman::Matrix R = Kalman::Matrix::Identity(3);
            R *= 4.0;                                            // sigma_z = 2 m per axis
            SetR(R);

            Kalman::Matrix P0 = Kalman::Matrix::Identity(6);
            P0 *= 100.;
            SetInitialCovariance(P0);
        }

        /// One step: gravity + quadratic drag, integrated over dt.
        Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &,
                         double = 0.) const override
        {
            const double dt = DeltaT();

            const double vx = x(3), vy = x(4), vz = x(5);
            const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);

            // a = -g z_hat - (c/m) |v| v
            const double ax = -fDragOverMass * speed * vx;
            const double ay = -fDragOverMass * speed * vy;
            const double az = -fGravity - fDragOverMass * speed * vz;

            Kalman::Vector out(6);
            // The 1-D position/velocity generator is nilpotent, so its exponential
            // is exactly [[1, dt], [0, 1]]: the closed form below is not an
            // approximation of a matrix exponential, it *is* the matrix exponential.
            out(0) = x(0) + dt * vx + 0.5 * ax * dt * dt;
            out(1) = x(1) + dt * vy + 0.5 * ay * dt * dt;
            out(2) = x(2) + dt * vz + 0.5 * az * dt * dt;
            out(3) = vx + dt * ax;
            out(4) = vy + dt * ay;
            out(5) = vz + dt * az;
            return out;
        }

        Kalman::Vector h(const Kalman::Vector &x, double = 0.) const override
        {
            Kalman::Vector z(3);
            z(0) = x(0); z(1) = x(1); z(2) = x(2);
            return z;
        }
};

int main()
{
    // --- the system ------------------------------------------------------
    Ballistics model(/* fs */ 50., /* c_drag/m */ 2e-3);

    Kalman::Vector x0(6);
    x0(0) = 0.;  x0(1) = 0.;  x0(2) = 0.;        // launch at the origin
    x0(3) = 60.; x0(4) = 15.; x0(5) = 90.;       // m/s
    model.SetInitialState(x0);

    Printf("%s", model.Describe().c_str());
    Printf("engine kind declared by the model : %s", Kalman::ToString(model.DefaultEngine()));

    // --- the data (truth + noisy position fixes) --------------------------
    Kalman::Sampler sampler(&model, 20260804);
    sampler.Generate(600, x0);

    // A radar that occasionally misses: NaN marks the gap, and the filter
    // coasts on prediction through it with no special case anywhere.
    sampler.DropChannel(2, 0.15);

    // --- the estimate -----------------------------------------------------
    Kalman::Filters::EKF filter(&model);
    const Kalman::Record forward = filter.Forward(sampler.Measurements());
    const Kalman::Record smoothed = filter.Backward();

    Printf("mean NIS  = %.3f   (consistent near m = %zu)", filter.MeanNIS(), model.NMeasure());
    Printf("mean NEES = %.3f   (consistent near n = %zu)", filter.MeanNEES(sampler.Truth()),
           model.NState());
    Printf("altitude RMSE : filtered = %.3f m, smoothed = %.3f m",
           forward.RMSE(2, sampler.Truth()), smoothed.RMSE(2, sampler.Truth()));

    // --- ROOT takes it from here -----------------------------------------
    gStyle->SetOptStat(0);
    TCanvas *c = new TCanvas("ballistics", "Ballistic tracking (EKF)", 1100, 620);
    c->SetGrid();

    const std::vector<double> t = forward.Times();
    const std::vector<double> zEst = forward.Series(2);       // altitude estimate
    const std::vector<double> zSig = forward.Sigma(2);        // its 1-sigma
    const std::vector<double> zSm = smoothed.Series(2);
    const std::vector<double> zMeas = forward.MeasurementSeries(2);
    std::vector<double> ex(forward.NSteps(), 0.);
    std::vector<double> zTruth(forward.NSteps());
    for (std::size_t k = 0; k < forward.NSteps(); ++k) zTruth[k] = sampler.Truth()[k](2);

    TMultiGraph *mg = new TMultiGraph();

    TGraphErrors *band = new TGraphErrors(forward.NSteps(), t.data(), zEst.data(),
                                          ex.data(), zSig.data());
    band->SetFillColorAlpha(kAzure - 9, 0.6);
    band->SetLineColor(kAzure + 2);
    band->SetLineWidth(2);
    mg->Add(band, "L3");

    // The dropped fixes are NaN; skip them rather than drawing a spike to zero.
    TGraph *gz = new TGraph();
    for (std::size_t k = 0; k < forward.NSteps(); ++k)
        if (!std::isnan(zMeas[k])) gz->SetPoint(gz->GetN(), t[k], zMeas[k]);
    gz->SetMarkerStyle(kFullCircle);
    gz->SetMarkerSize(0.4);
    gz->SetMarkerColor(kGray + 2);
    mg->Add(gz, "P");

    TGraph *gs = new TGraph(forward.NSteps(), t.data(), zSm.data());
    gs->SetLineColor(kRed + 1);
    gs->SetLineWidth(2);
    mg->Add(gs, "L");

    TGraph *gt = new TGraph(forward.NSteps(), t.data(), zTruth.data());
    gt->SetLineColor(kBlack);
    gt->SetLineStyle(kDashed);
    mg->Add(gt, "L");

    mg->SetTitle("Ballistic trajectory: 15% of altitude fixes dropped;time [s];altitude [m]");
    mg->Draw("A");

    // Bottom-right: the only corner the trajectory does not pass through. A
    // legend defaults to an opaque fill, so parking it over the curve silently
    // erases a stretch of the data and reads exactly like a gap in the input.
    TLegend *leg = new TLegend(0.62, 0.15, 0.88, 0.39);
    leg->SetFillStyle(0);
    leg->SetBorderSize(1);
    leg->AddEntry(gt, "truth", "l");
    leg->AddEntry(gz, "radar fix", "p");
    leg->AddEntry(band, "EKF #pm 1#sigma", "lf");
    leg->AddEntry(gs, "RTS smoother", "l");
    leg->Draw();

    c->SaveAs("ballistics_root.pdf");
    c->SaveAs("ballistics_root.png");   // handy for a README or a log

    TFile out("ballistics_root.root", "RECREATE");
    mg->Write("trajectory");
    out.Close();

    Printf("wrote ballistics_root.pdf, .png and .root");
    return 0;
}
