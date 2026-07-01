// main.C
// Individual fits (f0 FIXED, Q & Gi FREE; Q constrained to [mean±std]).
// Global fit (f0_i FIXED; Q_i & Gi_i FREE; Q_i constrained to [mean±std]).
// Individual pages show: green dashed seed, red individual, blue final (global).
// Saves: per-mode multipage individual PDF, one multi-pad global PDF, fits_out.root.

#include <TROOT.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TStyle.h>
#include <TMarker.h>
#include <TLegend.h>
#include <TMath.h>
#include <TH1D.h>
#include <TAxis.h>
#include <TFile.h>
#include <TDirectory.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <limits>

// ----------------- User knobs -----------------
static const std::string kFiQiFile = "fiQi.txt";
static const std::vector<std::string> kPsdFiles = {
    "K1/mode_1_psd.txt", "K1/mode_2_psd.txt", "K1/mode_3_psd.txt"
};
static const int  kModeIndex     = 0;
static const bool kFitAllModes   = true;

// frequency bands (Hz)
static const std::vector<std::pair<double,double>> kModeBands = {
    {170.0, 186.0},   // Mode 1
    {402.0, 432.0},   // Mode 2
    {718.0, 770.0}    // Mode 3
};

// Q constraints (mean ± std) per mode
static const std::vector<double> kQMean = {1.30e4, 1.80e4, 2.12e4};
static const std::vector<double> kQStd  = {2.00e3, 3.70e3, 6.90e3};

// max N peaks per mode (<=0 => use all)
static const int kNmax = -1;

// individual fit window and bounds
static const double kWinHz         = 1.0;
static const double kQMinFactor    = 0.5;   // legacy dynamic range; intersect with mean±std
static const double kQMaxFactor    = 2.0;
static const double kGiMaxFactor   = 100.0;

// global polish bounds (f0 FIXED globally; only Q & Gi free)
static const double kQPolishFactor  = 0.5;  // Q in [Qfit*0.5, Qfit*2.0] ∩ [mean±std]
static const double kGiPolishFactor = 5.0;  // Gi in [0, Gi_fit*5]

// ----------------- Helpers -----------------
template <typename T>
static inline T Clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo ? lo : (v > hi ? hi : v));
}

static inline bool InBand(double f, const std::pair<double,double>& band) {
    return (f >= band.first && f <= band.second);
}

static bool LoadFiQi(const std::string& path,
                     std::vector<double>& fi_all,
                     std::vector<double>& Qi_all)
{
    fi_all.clear(); Qi_all.clear();
    std::ifstream in(path);
    if (!in) { std::cerr << "ERROR: cannot open fiQi file: " << path << "\n"; return false; }
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        double f0, q0;
        if (ls >> f0 >> q0) {
            fi_all.push_back(f0);
            Qi_all.push_back(std::max(10.0, q0));
        }
    }
    return !(fi_all.empty() || fi_all.size()!=Qi_all.size());
}

static void LoadPSDBand(const std::string& psdFile, const std::pair<double,double>& band,
                        std::vector<double>& x, std::vector<double>& y)
{
    x.clear(); y.clear();
    std::ifstream in(psdFile);
    if (!in) { std::cerr << "Cannot open PSD file: " << psdFile << "\n"; return; }
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        double xf, yf;
        if (ls >> xf >> yf) {
            if (InBand(xf, band)) { x.push_back(xf); y.push_back(yf); }
        }
    }
}

static double NearestYAt(const std::vector<double>& x, const std::vector<double>& y, double f0) {
    if (x.empty()) return 1.0;
    size_t best = 0;
    double bd = std::abs(x[0]-f0);
    for (size_t i=1;i<x.size();++i) {
        double d = std::abs(x[i]-f0);
        if (d < bd) { bd = d; best = i; }
    }
    return std::max(0.0, y[best]);
}

// per-mode Q interval
static inline std::pair<double,double> ModeQInterval(int mode) {
    const double qlo = std::max(10.0, kQMean[mode] - kQStd[mode]);
    const double qhi = kQMean[mode] + kQStd[mode];
    return {qlo, qhi};
}

static inline std::pair<double,double> Intersect(double a1, double a2, double b1, double b2) {
    double lo = std::max(a1, b1);
    double hi = std::min(a2, b2);
    if (lo < hi) return {lo, hi};
    const double mid = 0.5*(b1+b2);
    const double eps = std::max(1.0, 0.001*mid);
    return {std::max(b1, mid-eps), std::min(b2, mid+eps)};
}

// ----------------- Models -----------------
static double OneLorentz(double* x, double* par) {
    const double f  = x[0];
    const double f0 = par[0];
    const double Q  = par[1];
    const double Gi = par[2];
    if (f0 <= 0.0 || Q <= 0.0 || Gi < 0.0) return 0.0;
    const double Gamma = f0 / Q; // FWHM
    const double df2   = (f*f - f0*f0);
    const double den   = df2*df2 + f*f*Gamma*Gamma;
    const double num   = (f0*f0) * (Gamma*Gamma);
    return (den > 0.0) ? (Gi * num / den) : 0.0;
}

static int g_Npeaks = 0;
static double SumLorentz(double* x, double* par) {
    const double f = x[0];
    double s = 0.0;
    for (int i=0;i<g_Npeaks;++i) {
        const double f0 = par[3*i+0];
        const double Q  = par[3*i+1];
        const double Gi = par[3*i+2];
        if (f0 <= 0.0 || Q <= 0.0 || Gi < 0.0) continue;
        const double Gamma = f0 / Q;
        const double df2   = (f*f - f0*f0);
        const double den   = df2*df2 + f*f*Gamma*Gamma;
        const double num   = (f0*f0) * (Gamma*Gamma);
        if (den > 0.0) s += Gi * num / den;
    }
    return s;
}

// ----------------- Per-peak book-keeping -----------------
struct PeakResult {
    double f0, f0e;
    double Q,  Qe;
    double Gi, Gie;
    double Gamma;
};

// Store handles so we can overlay final (global) curves on the same pages
struct PeakPage {
    TCanvas* c = nullptr;
    TH1D*    h = nullptr;
    TF1*     f_prefit = nullptr;   // [NEW] green dashed prefit
    TF1*     f_indiv  = nullptr;   // red individual fit
    double   f0 = 0, Q_indiv = 0, Gi_indiv = 0;
    // final (global) params will be drawn later
};

// ----------------- Mode driver -----------------
static void FitMode(int m, TFile* fout, TCanvas* cGlobalAll, int padIndex)
{
    const auto band = kModeBands[m];
    const std::string psdFile = kPsdFiles[m];

    TDirectory* dmode = fout->mkdir(Form("mode_%d", m+1));
    dmode->cd();

    // Load seeds
    std::vector<double> fi_all, Qi_all;
    if (!LoadFiQi(kFiQiFile, fi_all, Qi_all)) { std::cerr<<"Seed load failed.\n"; return; }

    // Seeds in band
    std::vector<std::pair<double,double>> seeds;
    for (size_t i=0;i<fi_all.size();++i) if (InBand(fi_all[i], band)) seeds.push_back({fi_all[i], Qi_all[i]});
    if (seeds.empty()) { std::cerr<<"No peaks in band "<<(m+1)<<"\n"; return; }
    if (kNmax>0 && (int)seeds.size()>kNmax) seeds.resize(kNmax);

    // Band PSD once
    std::vector<double> xband, yband;
    LoadPSDBand(psdFile, band, xband, yband);
    if (xband.size()<5) { std::cerr<<"Too few PSD points in band.\n"; return; }

    const auto [Qtable_lo, Qtable_hi] = ModeQInterval(m);

    // Prepare for multipage PDF (we will print AFTER global overlay)
    TString indivPdf = Form("mode_%d_individual_fits.pdf", m+1);
    bool firstPage = true;

    // Batch mode ON for individual pages (no windows pop up)
    const Bool_t prevBatch = gROOT->IsBatch();
    gROOT->SetBatch(kTRUE);

    // Accumulate results for global seeds and keep pages
    std::vector<PeakResult> results; results.reserve(seeds.size());
    std::vector<PeakPage>   pages;   pages.reserve(seeds.size());

    // ---------------- Individual fits: f0 FIXED, Q & Gi FREE ----------------
    for (size_t i=0;i<seeds.size(); ++i) {
        const double f0_seed = seeds[i].first;
        const double Q0_seed = seeds[i].second;

        // window data
        std::vector<double> xw, yw;
        for (size_t k=0;k<xband.size();++k)
            if (xband[k] >= f0_seed - kWinHz && xband[k] <= f0_seed + kWinHz)
                { xw.push_back(xband[k]); yw.push_back(yband[k]); }
        if (xw.size() < 5) { std::cerr<<"Peak "<<(i+1)<<": too few points.\n"; continue; }

        // histogram
        const int nbins = std::max<int>(50, std::min<int>(200, (int)xw.size()));
        auto* h = new TH1D(Form("h_m%d_p%zu", m+1, i+1),
                           Form("Mode %d - Peak %zu;Frequency [Hz];PSD [arb.]", m+1, i+1),
                           nbins, f0_seed - kWinHz, f0_seed + kWinHz);
        for (size_t k=0;k<xw.size();++k) {
            int bin = h->GetXaxis()->FindBin(xw[k]);
            h->AddBinContent(bin, yw[k]);
        }
        h->Write();

        // --- Prefit (seed) curve [NEW]: dashed green
        auto* f_prefit = new TF1(Form("f_prefit_m%d_p%zu", m+1, i+1), OneLorentz,
                                 f0_seed - kWinHz, f0_seed + kWinHz, 3);
        const double Gi0 = NearestYAt(xw, yw, f0_seed);
        // Q seed/bounds = (legacy dyn) ∩ (table)
        const double q_lo_dyn = std::max(10.0, Q0_seed*kQMinFactor);
        const double q_hi_dyn = std::max(100.0, Q0_seed*kQMaxFactor);
        const auto   q_int    = Intersect(q_lo_dyn, q_hi_dyn, Qtable_lo, Qtable_hi);
        const double q_seed   = Clamp(Q0_seed, q_int.first, q_int.second);

        f_prefit->SetParameter(0, f0_seed);
        f_prefit->FixParameter(0, f0_seed);
        f_prefit->SetParameter(1, q_seed);
        f_prefit->FixParameter(1, q_seed);
        f_prefit->SetParameter(2, Gi0);
        f_prefit->FixParameter(2, Gi0);
        f_prefit->SetNpx(1000);
        f_prefit->SetLineColor(kGreen+2);
        f_prefit->SetLineStyle(7);   // dashed
        f_prefit->SetLineWidth(3);

        // --- Individual fit (red solid)
        auto* f1 = new TF1(Form("f_one_m%d_p%zu", m+1, i+1), OneLorentz,
                           f0_seed - kWinHz, f0_seed + kWinHz, 3);
        f1->SetNpx(1000);
        f1->SetLineColor(kRed+1);
        f1->SetLineWidth(3);
        f1->SetParameter(0, f0_seed);
        f1->FixParameter(0, f0_seed);
        f1->SetParameter(1, q_seed);
        f1->SetParLimits(1, q_int.first, q_int.second);
        f1->SetParameter(2, Gi0);
        {
            double ymax = *std::max_element(yw.begin(), yw.end());
            f1->SetParLimits(2, 0.0, std::max(Gi0*kGiMaxFactor, ymax*2.0));
        }

        // canvas (we’ll draw prefit & fit now, but DO NOT print yet)
        auto* c = new TCanvas(Form("c_m%d_p%zu", m+1, i+1), "", 900, 500);
        c->SetLogy(true); gPad->SetGridx(); gPad->SetGridy();
        h->SetLineColor(kBlack); h->Draw("HIST");
        f_prefit->Draw("same"); // [NEW] seed overlay
        TFitResultPtr fr = h->Fit(f1, "RSMQ");
        f1->Draw("same");

        // marker at f0
        {
            const double yhat  = f1->Eval(f0_seed);
            auto* mark = new TMarker(f0_seed, yhat, 5);
            mark->SetMarkerColor(kRed+2);
            mark->SetMarkerSize(1.3);
            mark->Draw("same");
        }

        // results
        const double f0   = f1->GetParameter(0);  // fixed
        const double Q    = f1->GetParameter(1);
        const double Qe   = f1->GetParError(1);
        const double Gi   = f1->GetParameter(2);
        const double Ge   = f1->GetParError(2);
        const double Gamma= (Q>0.0 ? f0/Q : 0.0);
        std::cout << Form("Mode %d Peak %2zu: f0=%.6f (fixed), Q=%.6f ± %.6f (in [%.0f, %.0f]), Gi=%.6g ± %.6g, FWHM=%.6f Hz",
                          m+1, i+1, f0, Q, Qe, Qtable_lo, Qtable_hi, Gi, Ge, Gamma) << "\n";

        f1->Write();

        results.push_back(PeakResult{f0,0.0,Q,Qe,Gi,Ge,Gamma});
        pages.push_back(PeakPage{c,h,f_prefit,f1,f0,Q,Gi});
    }

    if (results.empty()) {
        std::cerr << "No successful individual fits in mode " << (m+1) << ". Aborting global fit.\n";
        return;
    }

    // ---------------- Global fit (f0 fixed), then overlay final (blue) on SAME pages ----------------
    g_Npeaks = (int)results.size();
    const int npar = 3 * g_Npeaks;

    // band graph
    auto* gr = new TGraph((int)xband.size(), xband.data(), yband.data());
    gr->SetName(Form("gr_mode%d_band", m+1));
    gr->SetTitle(Form("Mode %d - Global (f0 fixed; Q,Gi free; Q in mean±std)", m+1));
    gr->GetXaxis()->SetLimits(band.first, band.second);
    gr->SetLineColor(kBlack);
    gr->SetLineWidth(2);
    gr->Write();

    auto* fSum = new TF1(Form("f_sum_m%d", m+1), SumLorentz, band.first, band.second, npar);
    fSum->SetNpx(2000); fSum->SetLineColor(kRed+1); fSum->SetLineWidth(3); // red for global

    const double ymaxBand = *std::max_element(yband.begin(), yband.end());
    const auto [Qlo_tab, Qhi_tab] = ModeQInterval(m);
    for (int i=0;i<g_Npeaks;++i) {
        const int p_f0 = 3*i+0, p_Q = 3*i+1, p_G = 3*i+2;
        const double f0_fix = results[i].f0;
        const double Q0     = std::max(10.0, results[i].Q);
        const auto q_int = Intersect(std::max(10.0, Q0*kQPolishFactor),
                                     std::max(100.0, Q0/kQPolishFactor),
                                     Qlo_tab, Qhi_tab);
        const double q_seed = Clamp(Q0, q_int.first, q_int.second);
        const double Gi0    = std::max(0.0, results[i].Gi);

        fSum->SetParameter(p_f0, f0_fix); fSum->FixParameter(p_f0, f0_fix);
        fSum->SetParameter(p_Q,  q_seed); fSum->SetParLimits(p_Q, q_int.first, q_int.second);
        fSum->SetParameter(p_G,  Gi0);    fSum->SetParLimits(p_G, 0.0, std::max(Gi0*kGiPolishFactor, ymaxBand*2.0));
    }

    // draw global in shared canvas
    cGlobalAll->cd(padIndex);
    gPad->SetLogy(true); gPad->SetGridx(); gPad->SetGridy();
    gr->Draw("AL");
    std::cout << "\nMode " << (m+1) << " - Global fit (f0 fixed; Q constrained to mean±std):\n";
    TFitResultPtr frG = gr->Fit(fSum, "RSM");
    fSum->Draw("same");
    fSum->Write();

    // Extract final global params for each peak
    std::vector<double> Q_final(g_Npeaks), Gi_final(g_Npeaks);
    for (int i=0;i<g_Npeaks;++i) {
        Q_final[i]  = fSum->GetParameter(3*i+1);
        Gi_final[i] = fSum->GetParameter(3*i+2);
    }

    // -------- Overlay FINAL (global) on EACH individual page, then print PDF --------
    for (size_t i=0;i<pages.size(); ++i) {
        auto& P = pages[i];

        // Build final TF1 (blue solid) with (f0 fixed, Q=global, Gi=global)  [NEW]
        auto* f_final = new TF1(Form("f_final_m%d_p%zu", m+1, i+1), OneLorentz,
                                P.f0 - kWinHz, P.f0 + kWinHz, 3);
        f_final->SetNpx(1000);
        f_final->SetLineColor(kRed+1);
        f_final->SetLineWidth(3);
        f_final->SetParameter(0, P.f0);     f_final->FixParameter(0, P.f0);
        f_final->SetParameter(1, Q_final[i]); f_final->FixParameter(1, Q_final[i]);
        f_final->SetParameter(2, Gi_final[i]);f_final->FixParameter(2, Gi_final[i]);

        // Draw on SAME canvas (hist already drawn; prefit & indiv already drawn)
        P.c->cd();
        f_final->Draw("same");
        f_final->Write();

        // Legend [NEW]: show all 3
        auto* leg = new TLegend(0.70, 0.75, 0.88, 0.92);
        leg->AddEntry(P.h,        "PSD (histogram)", "l");
        leg->AddEntry(P.f_prefit, "Seed (pre-fit)",  "l");
        leg->AddEntry(P.f_indiv,  "Individual fit",  "l");
        leg->AddEntry(f_final,    "Global final",    "l");
        leg->Draw();

        // Print page now (after final overlay)
        if (firstPage) { P.c->Print(indivPdf + "(", "pdf"); firstPage = false; }
        else           { P.c->Print(indivPdf, "pdf"); }

        // Also persist the canvas
        P.c->Write();
    }

    // Close the multipage PDF
    if (!pages.empty()) {
        TCanvas dummyClose("dummy","",10,10);
        dummyClose.Print(indivPdf + ")", "pdf");
    }

    // Restore previous batch mode before global drawing
    gROOT->SetBatch(prevBatch);
    
    // Quick summary
    const double chi2 = frG ? frG->Chi2() : fSum->GetChisquare();
    const int    ndf  = frG ? frG->Ndf()  : fSum->GetNDF();
    std::cout << "  chi2 = " << chi2 << ",  ndf = " << ndf
              << ",  chi2/ndf = " << (ndf>0 ? chi2/ndf : -1.0) << "\n";
    for (int i=0;i<g_Npeaks;++i) {
        const double f0 = fSum->GetParameter(3*i+0);
        const double Q  = fSum->GetParameter(3*i+1);
        const double Gi = fSum->GetParameter(3*i+2);
        const double Gamma = (Q>0.0 ? f0/Q : 0.0);
        const auto [Qlo, Qhi] = ModeQInterval(m);
        std::cout << Form("  Peak %2d: f0=%.6f (fixed) Hz, Q=%.6f, Gi=%.6g, FWHM=%.6f Hz (Q in [%.0f, %.0f])",
                          i+1, f0, Q, Gi, Gamma, Qlo, Qhi) << "\n";
    }
}

// ----------------- Main -----------------
int main() {
    gStyle->SetOptStat(0);

    TFile* fout = TFile::Open("fits_out.root","RECREATE");
    if (!fout || fout->IsZombie()) { std::cerr << "ERROR: cannot create fits_out.root\n"; return 1; }

    std::vector<int> modes = kFitAllModes ? std::vector<int>{0,1,2} : std::vector<int>{kModeIndex};

    // Shared canvas for GLOBAL fits
    const int nModes = (int)modes.size();
    TCanvas* cGlobalAll = new TCanvas("cGlobalAll","Global fits (all modes)", 1500, 500*nModes);
    cGlobalAll->Divide(1, nModes);

    for (int i=0;i<nModes;++i) {
        fout->cd();
        FitMode(modes[i], fout, cGlobalAll, i+1);
    }

    cGlobalAll->SaveAs("global_fits_all_modes.pdf");
    fout->cd(); cGlobalAll->Write();
    fout->Write(); fout->Close();

    std::cout << "\nOutputs:\n"
              << "  - mode_<n>_individual_fits.pdf (now with seed (green dashed), individual (red), global (blue))\n"
              << "  - global_fits_all_modes.pdf\n"
              << "  - fits_out.root (all hists/TF1s/canvases)\n";
    return 0;
}