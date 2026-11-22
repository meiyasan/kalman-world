/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Kalman World", "index.html", [
    [ "What is inside", "index.html#what-is-inside", null ],
    [ "Quick start", "index.html#quick-start", null ],
    [ "Documentation", "index.html#documentation", null ],
    [ "License", "index.html#license", null ],
    [ "Applications", "applications_group.html", [
      [ "Applications at a glance", "md_docs_217___applications.html", null ],
      [ "Ballistics", "md_docs_2apps_2ballistics.html", [
        [ "The model", "md_docs_2apps_2ballistics.html#the-model", null ],
        [ "How it is built", "md_docs_2apps_2ballistics.html#how-it-is-built", null ],
        [ "Using it", "md_docs_2apps_2ballistics.html#using-it-1", null ]
      ] ],
      [ "Oscillators", "md_docs_2apps_2oscillators.html", [
        [ "The model", "md_docs_2apps_2oscillators.html#the-model-7", null ],
        [ "How it is built", "md_docs_2apps_2oscillators.html#how-it-is-built-7", null ],
        [ "Using it", "md_docs_2apps_2oscillators.html#using-it-9", null ],
        [ "Reconstruction: resampling artifacts and their fixes", "md_docs_2apps_2oscillators.html#reconstruction-resampling-artifacts-and-their-fixes", [
          [ "1. Asymmetric tail decay in the reconstructed time series", "md_docs_2apps_2oscillators.html#autotoc_md1-asymmetric-tail-decay-in-the-reconstructed-time-series", null ],
          [ "2. \"Bad suppression range\" — apparent leakage outside the window", "md_docs_2apps_2oscillators.html#autotoc_md2-bad-suppression-range--apparent-leakage-outside-the-window", null ]
        ] ]
      ] ],
      [ "Power grid", "md_docs_2apps_2powergrid.html", [
        [ "The model", "md_docs_2apps_2powergrid.html#the-model-8", null ],
        [ "How it is built", "md_docs_2apps_2powergrid.html#how-it-is-built-8", null ],
        [ "Using it", "md_docs_2apps_2powergrid.html#using-it-8", null ]
      ] ],
      [ "Suspension", "md_docs_2apps_2suspension.html", [
        [ "The model", "md_docs_2apps_2suspension.html#the-model-10", null ],
        [ "How it is built", "md_docs_2apps_2suspension.html#how-it-is-built-11", null ],
        [ "Using it", "md_docs_2apps_2suspension.html#using-it-13", null ]
      ] ],
      [ "Gravitational-wave polarization", "md_docs_2apps_2polarization.html", [
        [ "The model", "md_docs_2apps_2polarization.html#the-model-6", null ],
        [ "How it is built", "md_docs_2apps_2polarization.html#how-it-is-built-6", null ],
        [ "Using it", "md_docs_2apps_2polarization.html#using-it-7", null ]
      ] ],
      [ "Calorimeter", "md_docs_2apps_2calorimeter.html", [
        [ "The model", "md_docs_2apps_2calorimeter.html#the-model-2", null ],
        [ "How it is built", "md_docs_2apps_2calorimeter.html#how-it-is-built-2", null ],
        [ "Using it", "md_docs_2apps_2calorimeter.html#using-it-3", null ]
      ] ],
      [ "Rockets", "md_docs_2apps_2rocket.html", [
        [ "The model", "md_docs_2apps_2rocket.html#the-model-9", null ],
        [ "How it is built", "md_docs_2apps_2rocket.html#how-it-is-built-9", null ],
        [ "Using it", "md_docs_2apps_2rocket.html#using-it-10", null ]
      ] ],
      [ "Particle tracking", "md_docs_2apps_2tracking.html", [
        [ "The model", "md_docs_2apps_2tracking.html#the-model-11", null ],
        [ "How it is built", "md_docs_2apps_2tracking.html#how-it-is-built-12", null ],
        [ "Using it", "md_docs_2apps_2tracking.html#using-it-14", null ]
      ] ],
      [ "Space navigation", "md_docs_2apps_2space.html", [
        [ "The models", "md_docs_2apps_2space.html#the-models", null ],
        [ "Planning", "md_docs_2apps_2space.html#planning", null ],
        [ "How it is built", "md_docs_2apps_2space.html#how-it-is-built-10", null ],
        [ "Using it", "md_docs_2apps_2space.html#using-it-11", null ]
      ] ],
      [ "Sensor fusion", "md_docs_2apps_2fusion.html", [
        [ "The model", "md_docs_2apps_2fusion.html#the-model-3", null ],
        [ "How it is built", "md_docs_2apps_2fusion.html#how-it-is-built-3", null ],
        [ "Using it", "md_docs_2apps_2fusion.html#using-it-4", null ],
        [ "The same model, from operations", "md_docs_2apps_2fusion.html#the-same-model-from-operations", null ],
        [ "Generic sensors", "md_docs_2apps_2fusion.html#generic-sensors-1", null ]
      ] ],
      [ "Computer vision", "md_docs_2apps_2vision.html", [
        [ "The models", "md_docs_2apps_2vision.html#the-models-1", null ],
        [ "How it is built", "md_docs_2apps_2vision.html#how-it-is-built-13", null ],
        [ "Using it", "md_docs_2apps_2vision.html#using-it-16", null ]
      ] ],
      [ "GPS and relativity", "md_docs_2apps_2gps.html", [
        [ "The model", "md_docs_2apps_2gps.html#the-model-5", null ],
        [ "How it is built", "md_docs_2apps_2gps.html#how-it-is-built-5", null ],
        [ "Using it", "md_docs_2apps_2gps.html#using-it-6", null ]
      ] ],
      [ "Vehicles", "md_docs_2apps_2vehicles.html", [
        [ "The speed test", "md_docs_2apps_2vehicles.html#the-speed-test", null ],
        [ "Lane keeping", "md_docs_2apps_2vehicles.html#lane-keeping", null ],
        [ "How they are built", "md_docs_2apps_2vehicles.html#how-they-are-built-1", null ],
        [ "Using it", "md_docs_2apps_2vehicles.html#using-it-15", null ]
      ] ],
      [ "Turbulent flow", "md_docs_2apps_2flow.html", [
        [ "The model", "md_docs_2apps_2flow.html#the-model-4", null ],
        [ "Solver and integrator: two different jobs", "md_docs_2apps_2flow.html#solver-and-integrator-two-different-jobs", null ],
        [ "How it is built", "md_docs_2apps_2flow.html#how-it-is-built-4", null ],
        [ "Using it", "md_docs_2apps_2flow.html#using-it-5", null ]
      ] ],
      [ "Storm tracking", "md_docs_2apps_2storms.html", [
        [ "The storm model", "md_docs_2apps_2storms.html#the-storm-model", null ],
        [ "The tornado model", "md_docs_2apps_2storms.html#the-tornado-model", null ],
        [ "How they are built", "md_docs_2apps_2storms.html#how-they-are-built", null ],
        [ "Using it", "md_docs_2apps_2storms.html#using-it-12", null ]
      ] ]
    ] ],
    [ "User guide", "guide.html", [
      [ "Getting Started", "md_docs_21___getting-_started.html", [
        [ "1. A filter with no model class", "md_docs_21___getting-_started.html#autotoc_md1-a-filter-with-no-model-class", null ],
        [ "2. Get some data", "md_docs_21___getting-_started.html#autotoc_md2-get-some-data", null ],
        [ "3. Run the filter", "md_docs_21___getting-_started.html#autotoc_md3-run-the-filter", null ],
        [ "4. Look at it, and check it", "md_docs_21___getting-_started.html#autotoc_md4-look-at-it-and-check-it", null ],
        [ "The same thing with a named model", "md_docs_21___getting-_started.html#the-same-thing-with-a-named-model", null ],
        [ "Using a domain model instead", "md_docs_21___getting-_started.html#using-a-domain-model-instead", null ],
        [ "Growing a model with operations", "md_docs_21___getting-_started.html#growing-a-model-with-operations", null ],
        [ "Where to go next", "md_docs_21___getting-_started.html#where-to-go-next", null ]
      ] ],
      [ "Installation", "md_docs_22___installation.html", [
        [ "One command", "md_docs_22___installation.html#one-command", null ],
        [ "Manual install", "md_docs_22___installation.html#manual-install", [
          [ "Optional extras", "md_docs_22___installation.html#optional-extras", null ],
          [ "Conda (recommended with the C++ libraries)", "md_docs_22___installation.html#conda-recommended-with-the-c-libraries", null ]
        ] ],
        [ "Native acceleration (optional)", "md_docs_22___installation.html#native-acceleration-optional", null ],
        [ "Verify the install", "md_docs_22___installation.html#verify-the-install", null ],
        [ "Docker", "md_docs_22___installation.html#docker", null ],
        [ "Registering the MCP server with your AI client", "md_docs_22___installation.html#registering-the-mcp-server-with-your-ai-client", null ],
        [ "Layout after install", "md_docs_22___installation.html#layout-after-install", null ]
      ] ],
      [ "Overview", "md_docs_23___overview.html", [
        [ "The seven roles", "md_docs_23___overview.html#the-seven-roles-1", [
          [ "Integrator vs Solver", "md_docs_23___overview.html#integrator-vs-solver", null ]
        ] ],
        [ "Operations: a model made from models", "md_docs_23___overview.html#operations-a-model-made-from-models", null ],
        [ "Declared, not imported", "md_docs_23___overview.html#declared-not-imported", null ],
        [ "Notation", "md_docs_23___overview.html#notation", null ],
        [ "The pipeline", "md_docs_23___overview.html#the-pipeline", null ],
        [ "Package layout", "md_docs_23___overview.html#package-layout", null ],
        [ "One facade, three front doors", "md_docs_23___overview.html#one-facade-three-front-doors", null ]
      ] ],
      [ "Models", "md_docs_24___models.html", [
        [ "Path 1 — matrices", "md_docs_24___models.html#path-1--matrices", null ],
        [ "Path 2 — continuous", "md_docs_24___models.html#path-2--continuous", null ],
        [ "Path 3 — functions", "md_docs_24___models.html#path-3--functions", null ],
        [ "What a model may also define", "md_docs_24___models.html#what-a-model-may-also-define", null ],
        [ "Operations: a model made from models", "md_docs_24___models.html#operations-a-model-made-from-models-1", null ],
        [ "Composite models", "md_docs_24___models.html#composite-models", null ],
        [ "A system that changes regime: <tt>route</tt>", "md_docs_24___models.html#a-system-that-changes-regime-route", null ],
        [ "Joint models: estimating parameters with the state", "md_docs_24___models.html#joint-models-estimating-parameters-with-the-state", null ],
        [ "Sensors", "md_docs_24___models.html#sensors", [
          [ "Generic sensors", "md_docs_24___models.html#generic-sensors", null ],
          [ "Adapters: one piece of hardware", "md_docs_24___models.html#adapters-one-piece-of-hardware", null ],
          [ "Instruments", "md_docs_24___models.html#instruments", null ],
          [ "A sensor that reports late, and the lagged model", "md_docs_24___models.html#a-sensor-that-reports-late-and-the-lagged-model", null ]
        ] ],
        [ "The shipped catalog", "md_docs_24___models.html#the-shipped-catalog", null ],
        [ "Adding a model to the repository", "md_docs_24___models.html#adding-a-model-to-the-repository", null ],
        [ "Writing a model to a file", "md_docs_24___models.html#writing-a-model-to-a-file", null ]
      ] ],
      [ "Filters and Diagnostics", "md_docs_25___filters-and-_diagnostics.html", [
        [ "Choosing a filter", "md_docs_25___filters-and-_diagnostics.html#choosing-a-filter", [
          [ "The linear fast path", "md_docs_25___filters-and-_diagnostics.html#the-linear-fast-path", null ]
        ] ],
        [ "Running one", "md_docs_25___filters-and-_diagnostics.html#running-one", [
          [ "The steady-state gain", "md_docs_25___filters-and-_diagnostics.html#the-steady-state-gain", null ],
          [ "The record", "md_docs_25___filters-and-_diagnostics.html#the-record", null ]
        ] ],
        [ "The specialists", "md_docs_25___filters-and-_diagnostics.html#the-specialists", [
          [ "EnKF — high-dimensional assimilation", "md_docs_25___filters-and-_diagnostics.html#enkf--high-dimensional-assimilation", null ],
          [ "CKF — track finding", "md_docs_25___filters-and-_diagnostics.html#ckf--track-finding", null ],
          [ "Regime switching: <tt>route</tt>", "md_docs_25___filters-and-_diagnostics.html#regime-switching-route", null ]
        ] ],
        [ "Diagnostics — is the filter actually working?", "md_docs_25___filters-and-_diagnostics.html#diagnostics--is-the-filter-actually-working", [
          [ "Reading them", "md_docs_25___filters-and-_diagnostics.html#reading-them", null ],
          [ "A tuning trap worth naming", "md_docs_25___filters-and-_diagnostics.html#a-tuning-trap-worth-naming", null ]
        ] ]
      ] ],
      [ "Integrators, Solvers and Differentiators", "md_docs_26___integrators-and-_solvers.html", [
        [ "Integrators", "md_docs_26___integrators-and-_solvers.html#integrators", [
          [ "The contract", "md_docs_26___integrators-and-_solvers.html#the-contract", null ],
          [ "Shipped", "md_docs_26___integrators-and-_solvers.html#shipped", null ],
          [ "Choosing one", "md_docs_26___integrators-and-_solvers.html#choosing-one", null ],
          [ "Verifying it took effect", "md_docs_26___integrators-and-_solvers.html#verifying-it-took-effect", null ]
        ] ],
        [ "Solvers", "md_docs_26___integrators-and-_solvers.html#solvers", [
          [ "The contract", "md_docs_26___integrators-and-_solvers.html#the-contract-1", null ],
          [ "Shipped", "md_docs_26___integrators-and-_solvers.html#shipped-1", null ],
          [ "The Butcher-tableau protocol", "md_docs_26___integrators-and-_solvers.html#the-butcher-tableau-protocol", null ],
          [ "GPU / Torch backend", "md_docs_26___integrators-and-_solvers.html#gpu--torch-backend", null ]
        ] ],
        [ "Differentiators", "md_docs_26___integrators-and-_solvers.html#differentiators", null ],
        [ "Customization", "md_docs_26___integrators-and-_solvers.html#customization", [
          [ "1 — Swap in a different shipped scheme", "md_docs_26___integrators-and-_solvers.html#autotoc_md1--swap-in-a-different-shipped-scheme", null ],
          [ "2 — Subclass a shipped solver for your own physics", "md_docs_26___integrators-and-_solvers.html#autotoc_md2--subclass-a-shipped-solver-for-your-own-physics", null ],
          [ "3 — Write a scheme of your own", "md_docs_26___integrators-and-_solvers.html#autotoc_md3--write-a-scheme-of-your-own", null ],
          [ "4 — Register it under a name", "md_docs_26___integrators-and-_solvers.html#autotoc_md4--register-it-under-a-name", null ],
          [ "5 — Add a kind to the framework itself", "md_docs_26___integrators-and-_solvers.html#autotoc_md5--add-a-kind-to-the-framework-itself", null ]
        ] ],
        [ "Errors, not silent fallbacks", "md_docs_26___integrators-and-_solvers.html#errors-not-silent-fallbacks", null ],
        [ "Remaining rough edges", "md_docs_26___integrators-and-_solvers.html#remaining-rough-edges", null ],
        [ "Summary table", "md_docs_26___integrators-and-_solvers.html#summary-table", null ]
      ] ],
      [ "Providers", "md_docs_27___providers.html", [
        [ "The contract", "md_docs_27___providers.html#the-contract-2", null ],
        [ "Shipped providers", "md_docs_27___providers.html#shipped-providers", null ],
        [ "Resolving one", "md_docs_27___providers.html#resolving-one", null ],
        [ "By kind", "md_docs_27___providers.html#by-kind", [
          [ "Synthetic", "md_docs_27___providers.html#synthetic", null ],
          [ "File", "md_docs_27___providers.html#file", null ],
          [ "Sensor (streaming)", "md_docs_27___providers.html#sensor-streaming", null ],
          [ "Remote", "md_docs_27___providers.html#remote", null ],
          [ "ERA5 and turbulence", "md_docs_27___providers.html#era5-and-turbulence", null ]
        ] ],
        [ "Credentials", "md_docs_27___providers.html#credentials", null ],
        [ "Through the API", "md_docs_27___providers.html#through-the-api", [
          [ "Named sources", "md_docs_27___providers.html#named-sources", null ]
        ] ],
        [ "Writing your own", "md_docs_27___providers.html#writing-your-own", null ]
      ] ],
      [ "Control and Routing", "md_docs_28___control-and-_routing.html", [
        [ "Controllers", "md_docs_28___control-and-_routing.html#controllers", [
          [ "The contract", "md_docs_28___control-and-_routing.html#the-contract-3", null ],
          [ "Cost accounting", "md_docs_28___control-and-_routing.html#cost-accounting", null ],
          [ "Shipped controllers", "md_docs_28___control-and-_routing.html#shipped-controllers", null ],
          [ "Model predictive control", "md_docs_28___control-and-_routing.html#model-predictive-control", null ],
          [ "Tuners", "md_docs_28___control-and-_routing.html#tuners", null ],
          [ "Through the API", "md_docs_28___control-and-_routing.html#through-the-api-1", null ]
        ] ],
        [ "Routers", "md_docs_28___control-and-_routing.html#routers", null ],
        [ "Worked examples", "md_docs_28___control-and-_routing.html#worked-examples", null ]
      ] ],
      [ "The Kalman contract", "md_docs_219___contract.html", [
        [ "The rules", "md_docs_219___contract.html#the-rules", null ],
        [ "The model: what the system is", "md_docs_219___contract.html#the-model-what-the-system-is", null ],
        [ "The engine and the filter: how it is estimated", "md_docs_219___contract.html#the-engine-and-the-filter-how-it-is-estimated", null ],
        [ "Operations: a model made from another", "md_docs_219___contract.html#operations-a-model-made-from-another", null ],
        [ "Control: acting on the system", "md_docs_219___contract.html#control-acting-on-the-system", null ],
        [ "The three vocabularies", "md_docs_219___contract.html#the-three-vocabularies", [
          [ "Concepts every filter has", "md_docs_219___contract.html#concepts-every-filter-has", null ],
          [ "Concepts only one community has", "md_docs_219___contract.html#concepts-only-one-community-has", null ],
          [ "The operations, and what other fields call them", "md_docs_219___contract.html#the-operations-and-what-other-fields-call-them", null ],
          [ "Where each community's word lives in the code", "md_docs_219___contract.html#where-each-communitys-word-lives-in-the-code", null ]
        ] ],
        [ "Names being replaced", "md_docs_219___contract.html#names-being-replaced", null ],
        [ "Names made clearer", "md_docs_219___contract.html#names-made-clearer", null ]
      ] ]
    ] ],
    [ "API access", "access.html", [
      [ "Python API — the <tt>kalman.api</tt> facade", "md_docs_29___python-_a_p_i.html", [
        [ "The verbs", "md_docs_29___python-_a_p_i.html#the-verbs", null ],
        [ "<tt>KalmanSpec</tt> — describing a model as data", "md_docs_29___python-_a_p_i.html#kalmanspec--describing-a-model-as-data", [
          [ "The three bases", "md_docs_29___python-_a_p_i.html#the-three-bases", null ],
          [ "Bindings", "md_docs_29___python-_a_p_i.html#bindings", null ]
        ] ],
        [ "Worked session", "md_docs_29___python-_a_p_i.html#worked-session", [
          [ "Run IDs", "md_docs_29___python-_a_p_i.html#run-ids", null ]
        ] ],
        [ "Authoring: spec → registered model → reviewable code", "md_docs_29___python-_a_p_i.html#authoring-spec--registered-model--reviewable-code", null ],
        [ "OSSE and forecasting", "md_docs_29___python-_a_p_i.html#osse-and-forecasting", null ],
        [ "Adding a verb", "md_docs_29___python-_a_p_i.html#adding-a-verb", null ]
      ] ],
      [ "REST API", "md_docs_210___r_e_s_t-_a_p_i.html", [
        [ "Ports and addresses", "md_docs_210___r_e_s_t-_a_p_i.html#ports-and-addresses", [
          [ "Self-documenting", "md_docs_210___r_e_s_t-_a_p_i.html#self-documenting", null ]
        ] ],
        [ "Endpoints", "md_docs_210___r_e_s_t-_a_p_i.html#endpoints", null ],
        [ "Examples", "md_docs_210___r_e_s_t-_a_p_i.html#examples", [
          [ "Filtering a measurement set", "md_docs_210___r_e_s_t-_a_p_i.html#filtering-a-measurement-set", null ],
          [ "Follow-ups on a run", "md_docs_210___r_e_s_t-_a_p_i.html#follow-ups-on-a-run", null ],
          [ "Forecasting", "md_docs_210___r_e_s_t-_a_p_i.html#forecasting", null ],
          [ "Fetching from a provider", "md_docs_210___r_e_s_t-_a_p_i.html#fetching-from-a-provider", null ],
          [ "Authoring", "md_docs_210___r_e_s_t-_a_p_i.html#authoring", null ]
        ] ],
        [ "The shared registry", "md_docs_210___r_e_s_t-_a_p_i.html#the-shared-registry", [
          [ "Measurement sources", "md_docs_210___r_e_s_t-_a_p_i.html#measurement-sources", null ],
          [ "Subscribing to changes", "md_docs_210___r_e_s_t-_a_p_i.html#subscribing-to-changes", null ]
        ] ],
        [ "<tt>KALMAN_MAX_EVENTS</tt> events (default 10000).", "md_docs_210___r_e_s_t-_a_p_i.html#kalman_max_events-events-default-10000", null ],
        [ "Known rough edges", "md_docs_210___r_e_s_t-_a_p_i.html#known-rough-edges", null ],
        [ "Connecting from Unity", "md_docs_210___r_e_s_t-_a_p_i.html#connecting-from-unity", [
          [ "Sending measurements from a scene and getting the estimate back", "md_docs_210___r_e_s_t-_a_p_i.html#sending-measurements-from-a-scene-and-getting-the-estimate-back", null ]
        ] ]
      ] ],
      [ "MCP Server", "md_docs_211___m_c_p-_server.html", [
        [ "Tools", "md_docs_211___m_c_p-_server.html#tools", null ],
        [ "One registry for agents and HTTP clients", "md_docs_211___m_c_p-_server.html#one-registry-for-agents-and-http-clients", [
          [ "Change notifications", "md_docs_211___m_c_p-_server.html#change-notifications", null ]
        ] ],
        [ "Registering with a client", "md_docs_211___m_c_p-_server.html#registering-with-a-client", [
          [ "Per-user vs committed config", "md_docs_211___m_c_p-_server.html#per-user-vs-committed-config", null ]
        ] ],
        [ "Running it manually", "md_docs_211___m_c_p-_server.html#running-it-manually", [
          [ "SDK compatibility", "md_docs_211___m_c_p-_server.html#sdk-compatibility", null ]
        ] ],
        [ "What an agent can do with it", "md_docs_211___m_c_p-_server.html#what-an-agent-can-do-with-it", null ]
      ] ],
      [ "Authentication", "md_docs_212___authentication.html", [
        [ "Two layers", "md_docs_212___authentication.html#two-layers", null ],
        [ "Scopes", "md_docs_212___authentication.html#scopes", null ],
        [ "Enabling it", "md_docs_212___authentication.html#enabling-it", null ],
        [ "Creating an account", "md_docs_212___authentication.html#creating-an-account", null ],
        [ "Minting a token", "md_docs_212___authentication.html#minting-a-token", null ],
        [ "Logging in over HTTP", "md_docs_212___authentication.html#logging-in-over-http", null ],
        [ "Revoking a token", "md_docs_212___authentication.html#revoking-a-token", null ],
        [ "Presenting a token", "md_docs_212___authentication.html#presenting-a-token", null ],
        [ "Behaviour in detail", "md_docs_212___authentication.html#behaviour-in-detail", null ],
        [ "Practical setups", "md_docs_212___authentication.html#practical-setups", null ],
        [ "Rules", "md_docs_212___authentication.html#rules", null ]
      ] ],
      [ "Deployment", "md_docs_213___deployment.html", [
        [ "Ports", "md_docs_213___deployment.html#ports", null ],
        [ "Environment variables", "md_docs_213___deployment.html#environment-variables", null ],
        [ "Docker Compose", "md_docs_213___deployment.html#docker-compose", null ],
        [ "Running without Docker", "md_docs_213___deployment.html#running-without-docker", [
          [ "As a background service: <tt>make serve</tt>", "md_docs_213___deployment.html#as-a-background-service-make-serve", null ],
          [ "By hand", "md_docs_213___deployment.html#by-hand", null ],
          [ "One worker, please", "md_docs_213___deployment.html#one-worker-please", null ]
        ] ],
        [ "Hardening checklist", "md_docs_213___deployment.html#hardening-checklist", null ],
        [ "CI and the documentation site", "md_docs_213___deployment.html#ci-and-the-documentation-site", [
          [ "Versions", "md_docs_213___deployment.html#versions", null ],
          [ "What ends up in the site", "md_docs_213___deployment.html#what-ends-up-in-the-site", null ],
          [ "Building it yourself", "md_docs_213___deployment.html#building-it-yourself", null ]
        ] ]
      ] ],
      [ "Troubleshooting", "md_docs_214___troubleshooting.html", [
        [ "Import and packaging", "md_docs_214___troubleshooting.html#import-and-packaging", null ],
        [ "Model construction", "md_docs_214___troubleshooting.html#model-construction", null ],
        [ "Integrators and solvers", "md_docs_214___troubleshooting.html#integrators-and-solvers", null ],
        [ "Filtering results", "md_docs_214___troubleshooting.html#filtering-results", null ],
        [ "API and transports", "md_docs_214___troubleshooting.html#api-and-transports", null ],
        [ "Authentication", "md_docs_214___troubleshooting.html#authentication-1", null ],
        [ "Build and test", "md_docs_214___troubleshooting.html#build-and-test", null ]
      ] ]
    ] ],
    [ "C++ libraries", "cxx.html", [
      [ "libKalman — the C++ library", "md_docs_215___cxx-_library.html", [
        [ "Layout", "md_docs_215___cxx-_library.html#layout", [
          [ "Roles and utilities", "md_docs_215___cxx-_library.html#roles-and-utilities", null ]
        ] ],
        [ "The three roles", "md_docs_215___cxx-_library.html#the-three-roles", null ],
        [ "Building", "md_docs_215___cxx-_library.html#building", [
          [ "The four libraries", "md_docs_215___cxx-_library.html#the-four-libraries", null ],
          [ "The <tt>kalman</tt> program", "md_docs_215___cxx-_library.html#the-kalman-program", null ],
          [ "Optional backends", "md_docs_215___cxx-_library.html#optional-backends", null ],
          [ "The field solver on libTorch", "md_docs_215___cxx-_library.html#the-field-solver-on-libtorch", null ]
        ] ],
        [ "What to include", "md_docs_215___cxx-_library.html#what-to-include", null ],
        [ "Using it", "md_docs_215___cxx-_library.html#using-it", null ],
        [ "The decision layer", "md_docs_215___cxx-_library.html#the-decision-layer", [
          [ "Training in Python, deciding in C++", "md_docs_215___cxx-_library.html#training-in-python-deciding-in-c", null ]
        ] ],
        [ "The factory: how a C++ model reaches Python", "md_docs_215___cxx-_library.html#the-factory-how-a-c-model-reaches-python", [
          [ "Models that are their matrices", "md_docs_215___cxx-_library.html#models-that-are-their-matrices", null ],
          [ "Named arrays", "md_docs_215___cxx-_library.html#named-arrays", null ]
        ] ],
        [ "Model files: <tt>.kw</tt>", "md_docs_215___cxx-_library.html#model-files-kw", [
          [ "What a file can hold", "md_docs_215___cxx-_library.html#what-a-file-can-hold", null ],
          [ "The format", "md_docs_215___cxx-_library.html#the-format", null ],
          [ "Settings: <tt>name.kw.json</tt>", "md_docs_215___cxx-_library.html#settings-namekwjson", null ],
          [ "What is checked", "md_docs_215___cxx-_library.html#what-is-checked", null ],
          [ "From C++, C and Python", "md_docs_215___cxx-_library.html#from-c-c-and-python", null ],
          [ "From Python", "md_docs_215___cxx-_library.html#from-python", null ]
        ] ],
        [ "libKalmanApps: the application models in C++", "md_docs_215___cxx-_library.html#libkalmanapps-the-application-models-in-c", [
          [ "Field models on the ensemble engine", "md_docs_215___cxx-_library.html#field-models-on-the-ensemble-engine", null ],
          [ "Joint models: the counterpart, wrapped", "md_docs_215___cxx-_library.html#joint-models-the-counterpart-wrapped", null ],
          [ "Observability", "md_docs_215___cxx-_library.html#observability", null ],
          [ "Sensors, and a model read by one more", "md_docs_215___cxx-_library.html#sensors-and-a-model-read-by-one-more", null ],
          [ "Composite models", "md_docs_215___cxx-_library.html#composite-models-1", null ]
        ] ],
        [ "Integrating with root-plus", "md_docs_215___cxx-_library.html#integrating-with-root-plus", [
          [ "The ROOT+ side", "md_docs_215___cxx-_library.html#the-root-side", null ],
          [ "Linking it yourself", "md_docs_215___cxx-_library.html#linking-it-yourself", null ]
        ] ],
        [ "Differences from the Python framework", "md_docs_215___cxx-_library.html#differences-from-the-python-framework", null ],
        [ "Where the Python boundary is, and why", "md_docs_215___cxx-_library.html#where-the-python-boundary-is-and-why", null ]
      ] ],
      [ "C++ models from Python", "md_docs_216___cxx-_models-from-_python.html", [
        [ "What runs where", "md_docs_216___cxx-_models-from-_python.html#what-runs-where", null ],
        [ "Controllers and routers", "md_docs_216___cxx-_models-from-_python.html#controllers-and-routers", null ],
        [ "Install and check", "md_docs_216___cxx-_models-from-_python.html#install-and-check", null ],
        [ "Three ways to write a model", "md_docs_216___cxx-_models-from-_python.html#three-ways-to-write-a-model", null ],
        [ "The toy: one model, two languages", "md_docs_216___cxx-_models-from-_python.html#the-toy-one-model-two-languages", [
          [ "The physics", "md_docs_216___cxx-_models-from-_python.html#the-physics", null ],
          [ "The model, in C++", "md_docs_216___cxx-_models-from-_python.html#the-model-in-c", null ],
          [ "Using it from C++", "md_docs_216___cxx-_models-from-_python.html#using-it-from-c", null ],
          [ "Using it from Python", "md_docs_216___cxx-_models-from-_python.html#using-it-from-python", null ],
          [ "The model, in Python", "md_docs_216___cxx-_models-from-_python.html#the-model-in-python", null ],
          [ "What you should see", "md_docs_216___cxx-_models-from-_python.html#what-you-should-see", null ]
        ] ],
        [ "Named parameters", "md_docs_216___cxx-_models-from-_python.html#named-parameters", null ],
        [ "Giving an existing Python model a C++ counterpart", "md_docs_216___cxx-_models-from-_python.html#giving-an-existing-python-model-a-c-counterpart", null ],
        [ "When something goes wrong", "md_docs_216___cxx-_models-from-_python.html#when-something-goes-wrong", null ]
      ] ],
      [ "libKalman — the Kalman framework in C++", "libkalman.html", [
        [ "The three roles", "libkalman.html#libkalman_roles", null ],
        [ "The decision layer", "libkalman.html#libkalman_routing", null ],
        [ "No applications here", "libkalman.html#libkalman_noapps", null ],
        [ "Optional backends", "libkalman.html#libkalman_optional", null ]
      ] ],
      [ "libKalmanApps — the application models", "libkalmanapps.html", null ],
      [ "libKalmanLearn — learned components", "libkalmanlearn.html", null ],
      [ "libKalmanNet: the JSON API, REST and MCP servers in C++", "md_docs_218___net.html", [
        [ "What it serves, and what stays Python", "md_docs_218___net.html#what-it-serves-and-what-stays-python", null ],
        [ "Build", "md_docs_218___net.html#build", null ],
        [ "Start it", "md_docs_218___net.html#start-it", null ],
        [ "Routes and tools", "md_docs_218___net.html#routes-and-tools", null ],
        [ "Tokens", "md_docs_218___net.html#tokens", null ],
        [ "From Python, in process", "md_docs_218___net.html#from-python-in-process", null ],
        [ "Agreement with Python, and speed", "md_docs_218___net.html#agreement-with-python-and-speed", null ],
        [ "Rules for changing it", "md_docs_218___net.html#rules-for-changing-it", null ]
      ] ]
    ] ],
    [ "For developers", "developers.html", [
      [ "AGENTS.md — working on this codebase with an LLM", "md__a_g_e_n_t_s.html", [
        [ "1. What this project is", "md__a_g_e_n_t_s.html#autotoc_md1-what-this-project-is", [
          [ "The seven roles", "md__a_g_e_n_t_s.html#the-seven-roles", null ]
        ] ],
        [ "2. Non-negotiable invariants", "md__a_g_e_n_t_s.html#autotoc_md2-non-negotiable-invariants", null ],
        [ "3. Layout", "md__a_g_e_n_t_s.html#autotoc_md3-layout", null ],
        [ "4. How to add things", "md__a_g_e_n_t_s.html#autotoc_md4-how-to-add-things", null ],
        [ "5. Numerical conventions that bite", "md__a_g_e_n_t_s.html#autotoc_md5-numerical-conventions-that-bite", null ],
        [ "6. Verification — required before claiming done", "md__a_g_e_n_t_s.html#autotoc_md6-verification--required-before-claiming-done", null ],
        [ "7. Working style expected here", "md__a_g_e_n_t_s.html#autotoc_md7-working-style-expected-here", null ],
        [ "8. Known deferred work", "md__a_g_e_n_t_s.html#autotoc_md8-known-deferred-work", null ],
        [ "9. Documentation", "md__a_g_e_n_t_s.html#autotoc_md9-documentation", null ],
        [ "10. libKalman — the C++ port", "md__a_g_e_n_t_s.html#autotoc_md10-libkalman--the-c-port", null ]
      ] ],
      [ "Contributing", "md__c_o_n_t_r_i_b_u_t_i_n_g.html", null ]
    ] ],
    [ "CO2 in a room", "md_docs_2apps_2buildings.html", [
      [ "The model", "md_docs_2apps_2buildings.html#the-model-1", null ],
      [ "How it is built", "md_docs_2apps_2buildings.html#how-it-is-built-1", null ],
      [ "Using it", "md_docs_2apps_2buildings.html#using-it-2", null ],
      [ "Windows, doors and air conditioning", "md_docs_2apps_2buildings.html#windows-doors-and-air-conditioning", null ],
      [ "A known empty room", "md_docs_2apps_2buildings.html#a-known-empty-room", null ],
      [ "Real sensors: Netatmo", "md_docs_2apps_2buildings.html#real-sensors-netatmo", null ]
    ] ],
    [ "Kalman World documentation", "md_docs_2_r_e_a_d_m_e.html", [
      [ "For users", "md_docs_2_r_e_a_d_m_e.html#for-users", null ],
      [ "For API access", "md_docs_2_r_e_a_d_m_e.html#for-api-access", null ],
      [ "For C++ / ROOT", "md_docs_2_r_e_a_d_m_e.html#for-c--root", null ],
      [ "The applications", "md_docs_2_r_e_a_d_m_e.html#the-applications", null ],
      [ "Also in this repository", "md_docs_2_r_e_a_d_m_e.html#also-in-this-repository", null ],
      [ "Conventions used in these pages", "md_docs_2_r_e_a_d_m_e.html#conventions-used-in-these-pages", null ]
    ] ],
    [ "Notebooks", "notebooks/index.html", null ],
    [ "API Reference", "usergroup0.html", [
      [ "Namespaces", "namespaces.html", [
        [ "Namespace List", "namespaces.html", "namespaces_dup" ],
        [ "Namespace Members", "namespacemembers.html", [
          [ "All", "namespacemembers.html", "namespacemembers_dup" ],
          [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
          [ "Variables", "namespacemembers_vars.html", "namespacemembers_vars" ],
          [ "Typedefs", "namespacemembers_type.html", null ],
          [ "Enumerations", "namespacemembers_enum.html", null ]
        ] ]
      ] ],
      [ "Classes", "annotated.html", [
        [ "Class List", "annotated.html", "annotated_dup" ],
        [ "Class Index", "classes.html", null ],
        [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
        [ "Class Members", "functions.html", [
          [ "All", "functions.html", "functions_dup" ],
          [ "Functions", "functions_func.html", "functions_func" ],
          [ "Variables", "functions_vars.html", "functions_vars" ],
          [ "Typedefs", "functions_type.html", null ],
          [ "Enumerations", "functions_enum.html", null ],
          [ "Related Symbols", "functions_rela.html", null ]
        ] ]
      ] ],
      [ "Files", "files.html", [
        [ "File List", "files.html", "files_dup" ],
        [ "File Members", "globals.html", [
          [ "All", "globals.html", "globals_dup" ],
          [ "Functions", "globals_func.html", "globals_func" ],
          [ "Variables", "globals_vars.html", null ],
          [ "Typedefs", "globals_type.html", null ],
          [ "Macros", "globals_defs.html", null ]
        ] ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"00__utils_8cc.html",
"_c_a_p_i_8cc.html#aaf368f4d2de5dc09d12a8e915cd4ffdb",
"_f_f_t_8h.html#a29ca70cfb40f375d5a9fae24e2a47f6d",
"_sampler_8h_source.html",
"ballistics__root_8cc_source.html",
"class_kalman_1_1_apps_1_1_ballistics_1_1_extended_model.html#af6941b4df31b3935d1a2289f6249440b",
"class_kalman_1_1_controller.html#a3ababe8434b98164fff512789298e6b3",
"class_kalman_1_1_differentiators_1_1_forward.html",
"class_kalman_1_1_engines_1_1_particle.html#af2e696c4a48f0a54e171408dad928d07",
"class_kalman_1_1_filters_1_1_p_f.html",
"class_kalman_1_1_integrator.html#a2efdda2247360782e3410675d49a4a25",
"class_kalman_1_1_model.html#a74d8fb2fe3f798744296d2c4ad1f2a01",
"class_kalman_1_1_models_1_1_composite.html#a7d76aa80038e105cc50cdf53d9590d5b",
"class_kalman_1_1_models_1_1_lag.html#afd2bd4c9bbbba370b3c08f6c2a87ef7d",
"class_kalman_1_1_routers_1_1_n_n_1_1_m_l_p.html#a77921079ab0a277ddcb62b165b015b63",
"class_kalman_1_1_solvers_1_1_finite_difference.html#ae8d8efe1719e0fc627a2baea13fb72fd",
"classmodules_1_1adapters_1_1netatmo_1_1_netatmo_cloud.html#a6f09aacd160e72d47d21c7878e7fec43",
"classmodules_1_1apps_1_1aerospace_1_1landing_1_1_rocket_landing_model.html#aef03a922c8bf8589bab3f29589012228",
"classmodules_1_1apps_1_1astrophysics_1_1translunar_1_1_trans_lunar_model.html#afdea48743df76ae9f5d1c42ccd9150a0",
"classmodules_1_1apps_1_1calorimeter_1_1models_1_1___calo_energy_model.html#a082f1b8d94e92efbc3a82c2c15d93566",
"classmodules_1_1apps_1_1gps_1_1model_1_1_g_p_s_model.html#a90e5edc9103f98c5e09590c1224cca23",
"classmodules_1_1apps_1_1resonators_1_1oscillators_1_1composite_1_1_oscillator_model.html#ac333ea0d72c7d3f593b82de774ef5e16",
"classmodules_1_1apps_1_1vehicles_1_1speed_1_1_car_speed_model.html#ad49df15c3cc98d06ea5f9657e317e512",
"classmodules_1_1apps_1_1weather_1_1flow_1_1_twin_experiment.html#a0b229c0acff944cee3c1df1df205da14",
"classmodules_1_1framework_1_1abstract_1_1contracts_1_1_kalman_engine_a_b_c.html#a14507b86b28511f546641f2f8321994d",
"classmodules_1_1framework_1_1abstract_1_1mixins_1_1_kalman_run_mixin.html#a2807efad4f91de6a6d06abc80d724068",
"classmodules_1_1framework_1_1controllers_1_1mppi_1_1_m_p_p_i_kalman_controller.html#ae393ac01c3ac6e32923220b6777d1f8a",
"classmodules_1_1framework_1_1engines_1_1ensemble_1_1_ensemble_kalman_engine.html#afa252afec6e748ca2101fb879545c4f3",
"classmodules_1_1framework_1_1filters_1_1ckf_1_1_combinatorial_kalman_filter.html#adb54af3561773945e9bbee3bb1ae9c30",
"classmodules_1_1framework_1_1model_1_1_kalman_model.html#a13879d96f94541e703166208315066bf",
"classmodules_1_1framework_1_1models_1_1joint_1_1_joint.html#ab8eff234c50652f897df7b97d851ceaf",
"classmodules_1_1framework_1_1routers_1_1base_1_1_kalman_router.html#a38ef5212cfc74861ba68cdfe2476b143",
"classmodules_1_1framework_1_1sampler_1_1_kalman_sampler.html#a8871645e0c30b3ae2b345000c32d4fb7",
"classmodules_1_1framework_1_1solvers_1_1spectral_1_1_spectral_solver.html#abf452af7887d861be9801adc596f4781",
"classmodules_1_1mcp_1_1watch_1_1_registry_watcher.html#a230ee164b6287f298555a91e8a89f954",
"classtests_1_1test__add__sensor_1_1_test_ballistic_doppler.html#af7fc1b82e4196279f63c21da09264051",
"classtests_1_1test__apps__extra_1_1_test_rocket_landing.html#ad56dc64fa2e9c26558f24f2c6c2434e5",
"classtests_1_1test__ensemble_1_1_test_field_gradients.html#adad9d88080e962a8206a29e7e1dc3bd6",
"classtests_1_1test__framework_1_1_test_sampler.html",
"classtests_1_1test__interferometer_1_1_test_detector_subclasses.html#aa538a1f586cd0045fa44abb6eb822191",
"classtests_1_1test__module_1_1_test_model_dimensions.html#ab4f4d7d97d5c34abfc1e4c879cd4f10f",
"classtests_1_1test__native__composite_1_1_test_sampling_matrices.html#a325fd57b6b4618135d7ea1cd32ccb371",
"classtests_1_1test__native__counterparts_1_1_test_car_lane_counterpart.html#aea48a96d3576ae3607a7e87830489b7b",
"classtests_1_1test__native__matrices_1_1_test_calorimeter.html#a3617b1acb36ebacdf5a4a917e5efcd05",
"classtests_1_1test__oscillators_1_1_test_analytic_oscillator_model.html#af221f9c53446e226c3c8c59bef6b645d",
"classtests_1_1test__registry_1_1_test_login_and_revocation.html#ac88d1e2f349d31aee578f1e7f6989f3f",
"classtests_1_1test__routers_1_1_test_bayesian_router.html#a176133a219b7adb2a5eb5c05662fe7a7",
"classtests_1_1test__storms_1_1_test_typhoon_tracking.html#a5d5af7276e76bfad0b76a4fa1ce62a36",
"cxx.html",
"framework_2models_2native_8py.html#a02e0994c4f10512effe10641ee7918a4",
"include_2_kalman_2_c_a_p_i_8h.html#a13a5750a06b2d9cdbedf33885435d7d5",
"learn_2include_2_kalman_2_learn_2_c_a_p_i_8h.html",
"md_docs_25___filters-and-_diagnostics.html#running-one",
"modules_2apps_2resonators_2____init_____8py_source.html",
"namespace_kalman_1_1_routers_1_1_n_n.html",
"namespacemodules_1_1api_1_1store.html#ad181dea0b5696489bdb9c5633afb6972",
"namespacemodules_1_1apps_1_1weather_1_1geo.html#a06427d7b2895858011ccff3e11f989ac",
"namespacemodules_1_1framework_1_1native_1_1learn.html#ad8b1d3cc0643c9a0bef8b5b0ef372eca",
"namespacetests_1_1test__barotropic.html",
"namespacetests_1_1test__registry.html#a3929f81ba8ea27092a2d65c8b5ec7888",
"struct_fixture_1_1_tracking_event.html#a50c3c14aa49d2a2ffe937ee924950d74",
"struct_kalman_1_1_tableau.html",
"test__kalman__cli_8py.html#a68f88bad7ce1f9924dfcf2c434cdccbc",
"toy__model_8py_source.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';