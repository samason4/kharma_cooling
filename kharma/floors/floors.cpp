/* 
 *  File: floors.cpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "floors.hpp"
#include "floors_functions.hpp"
#include "floors_impl.hpp"

#include "domain.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"

// Floors.  Apply limits to fluid values to maintain integrable state

int Floors::CountFFlags(MeshData<Real> *md)
{
    return Reductions::CountFlags(md, "fflag", FFlag::flag_names, IndexDomain::interior, true)[0];
}

std::shared_ptr<KHARMAPackage> Floors::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Floors");
    Params &params = pkg->AllParams();

    // Parse all the particular floor values into a nice struct we can pass device-side
    params.Add("prescription", Floors::Prescription(pin));

    // Frame to apply floors: usually we use normal observer frame, but
    // the option exists to use the fluid frame exclusively 'fluid' or outside a
    // certain radius 'mixed'. This option adds fluid at speed, making results
    // less reliable but velocity reconstructions potentially more robust.
    // Drift frame floors are now available and preferred when using 
    // the implicit solver to avoid UtoP calls.
    // TODO(BSP) automate/standardize parsing enums like this: classes w/tables like the flags?
    std::vector<std::string> allowed_floor_frames = {"normal", "fluid",
                                                     "mixed", "drift"};
    std::string frame_s = pin->GetOrAddString("floors", "frame", "drift", allowed_floor_frames);
    InjectionFrame frame;
    if (frame_s == "normal") {
        frame = InjectionFrame::normal;
    } else if (frame_s == "fluid") {
        frame = InjectionFrame::fluid;
    } else if (frame_s == "mixed") {
        frame = InjectionFrame::mixed;
    } else if (frame_s == "drift") {
        frame = InjectionFrame::drift;
    }
    params.Add("frame", frame);

    // Disable all floors.  It is obviously tremendously inadvisable to
    // set this option to true
    bool disable_floors = pin->GetOrAddBoolean("floors", "disable_floors", false);
    params.Add("disable_floors", disable_floors);

    // Flag for which floor conditions were violated.  Used for diagnostics
    // TODO(BSP) Should switch these to "Integer" fields when Parthenon supports it
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("fflag", m);
    // When not using UtoP, we still need a dummy copy of pflag, too
    // TODO we shouldn't require pflag
    if (!packages->AllPackages().count("Inverter")) {
        pkg->AddField("pflag", m);
    }

    // These preserve floor values between the "mark" pass and the actual floor application
    // They exist because sometimes we need to mark zones which need floors,
    // but then do e.g. FOFC to them instead in the hope that they will stop needing actual floors
    pkg->AddField("Floors.rho_floor", m);
    pkg->AddField("Floors.u_floor", m);

    // TODO(BSP) THIS IS THE ONLY MeshApplyFloors.  Any others will NOT BE CALLED.
    // Use BlockApplyFloors in your packages or fix Packages::MeshApplyFloors
    pkg->MeshApplyFloors = Floors::ApplyGRMHDFloors;
    pkg->PostStepDiagnosticsMesh = Floors::PostStepDiagnostics;

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Count total floors as a history item
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, CountFFlags, "FFlags"));
    // TODO Domain::entire version?
    // TODO entries for each individual flag?
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus Floors::ApplyInitialFloors(ParameterInput *pin, MeshBlockData<Real> *mbd, IndexDomain domain)
{
    Flag("ApplyInitialFloors");

    auto pmb = mbd->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = mbd->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto U = mbd->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto floor_vals = mbd->PackVariables(std::vector<std::string>{"Floors.rho_floor", "Floors.u_floor"});

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // If we're going to apply floors through the run, apply the same ones at init
    // Otherwise stick to specified/default geometric floors
    Floors::Prescription floors_tmp;
    if (pmb->packages.AllPackages().count("Floors")) {
        floors_tmp = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    } else {
            // JUST rho & u geometric
            floors_tmp.rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1e-6);
            floors_tmp.u_min_geom   = pin->GetOrAddReal("floors", "u_min_geom", 1e-8);

            // Disable everything else, even if it's specified
            floors_tmp.bsq_over_rho_max = 1e20;
            floors_tmp.bsq_over_u_max   = 1e20;
            floors_tmp.u_over_rho_max   = 1e20;
            floors_tmp.ktot_max         = 1e20;
            floors_tmp.gamma_max        = 1e20;

            floors_tmp.use_r_char    = false;
            floors_tmp.r_char        = 0.; //unused
            floors_tmp.temp_adjust_u = false;
            floors_tmp.adjust_k      = false;
    }
    const Floors::Prescription floors = floors_tmp;

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_initial_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            determine_floors(G, P, m_p, gam, k, j, i, floors, floor_vals);
            apply_floors<InjectionFrame::fluid>(G, P, m_p, gam, emhd_params, k, j, i, floor_vals, U, m_u);
            apply_ceilings(G, P, m_p, gam, k, j, i, floors, U, m_u);
        }
    );

    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Floors::DetermineGRMHDFloors(MeshData<Real> *md, IndexDomain domain, const Floors::Prescription& floors)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Packs of prims and cons
    PackIndexMap prims_map;
    auto& P = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = md->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = md->PackVariables(std::vector<std::string>{"pflag"});
    auto floor_vals = md->PackVariables(std::vector<std::string>{"Floors.rho_floor", "Floors.u_floor"});

    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    const IndexRange3 b = KDomain::GetRange(md, domain);
    const IndexRange block = IndexRange{0, P.GetDim(5) - 1};
    pmb0->par_for("determine_floors", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            // Flag & apply floors for successful and failed zones, just not corners
            // TODO(BSP) still necessary?
            if (((int) pflag(b, 0, k, j, i)) >= (int) Inverter::Status::success) {
                const auto& G = P.GetCoords(b);
                fflag(b, 0, k, j, i) = determine_floors(G, P(b), m_p, gam, k, j, i, floors, floor_vals(b));
            }
        }
    );

    // TODO(BSP) if we can somehow guarantee one call/rank we can start the reduction here
    //Reductions::StartFlagReduce(md, "fflag", FFlag::flag_names, IndexDomain::interior, true, 0);

    return TaskStatus::complete;
}

TaskStatus Floors::ApplyGRMHDFloors(MeshData<Real> *md, IndexDomain domain)
{
    auto pmesh = md->GetMeshPointer();
    const auto& pars = pmesh->packages.Get("Floors")->AllParams();
    if (pars.Get<InjectionFrame>("frame") == InjectionFrame::normal) {
        return ApplyFloorsInFrame<InjectionFrame::normal>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::fluid) {
        return ApplyFloorsInFrame<InjectionFrame::fluid>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::mixed) {
        return ApplyFloorsInFrame<InjectionFrame::mixed>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::drift) {
        return ApplyFloorsInFrame<InjectionFrame::drift>(md, domain);
    } else {
        throw std::invalid_argument("Floors for requested frame not implemented!");
    }
}

TaskStatus Floors::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about floor flags
    if (flag_verbose > 0) {
        Reductions::StartFlagReduce(md, "fflag", FFlag::flag_names, IndexDomain::interior, true, 0);
        // Debugging/diagnostic info about floor and inversion flags
        Reductions::CheckFlagReduceAndPrintHits(md, "fflag", FFlag::flag_names, IndexDomain::interior, true, 0);
    }

    // Anything else (energy conservation? Added material stats?)

    return TaskStatus::complete;
}
