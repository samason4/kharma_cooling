/* 
 *  File: conducting_atmosphere.cpp
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

#include "emhd/conducting_atmosphere.hpp"

#include "b_flux_ct.hpp"
#include "boundaries.hpp"
#include "coordinate_utils.hpp"

using namespace parthenon;

#define STRLEN 2048

/**
 * Initialization of the hydrostatic conducting atmosphere test
 * 
 * The ODE solution (kharma/prob/emhd/conducting_atmosphere_${RES}_default) is the input to the code.
 * Since the ODE solution is a steady-state solution of the EMHD equations,
 * the code should maintain the solution.
 */
TaskStatus InitializeAtmosphere(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    // Obtain EMHD params
    const bool use_emhd     = pmb->packages.AllPackages().count("EMHD");
    EMHD::EMHD_parameters emhd_params = EMHD::GetEMHDParameters(pmb->packages);
    emhd_params.higher_order_terms = false;

    // Obtain GRMHD params
    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real& gam        = grmhd_pars.Get<Real>("gamma");

    // Get all primitive variables (GRMHD+EMHD if in use)
    PackIndexMap prims_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    VarMap m_p(prims_map, false);

    const auto& G = pmb->coords;

    // Type of input to the problem
    const std::string input = pin->GetOrAddString("conducting_atmosphere", "input", "ODE");

    // Bounds of the domain
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    // Load file names into strings
    // TODO store as single file table. HDF5?
    char fode_rCoords[STRLEN], fode_rho[STRLEN], fode_u[STRLEN], fode_q[STRLEN];
    sprintf(fode_rCoords, "atmosphere_soln_rCoords.txt");
    sprintf(fode_rho,     "atmosphere_soln_rho.txt");
    sprintf(fode_u,       "atmosphere_soln_u.txt");
    sprintf(fode_q,       "atmosphere_soln_phi.txt");

    // Assign file pointers
    FILE *fp_r, *fp_rho, *fp_u, *fp_q;;
    fp_r   = fopen(fode_rCoords, "r");
    fp_rho = fopen(fode_rho, "r");
    fp_u   = fopen(fode_u,   "r");
    if (fp_r == NULL || fp_rho == NULL || fp_u == NULL) {
        throw std::runtime_error("Could not open conducting atmosphere solution!");
    }
    if (use_emhd) {
        fp_q = fopen(fode_q, "r");
        if (fp_q == NULL) {
            throw std::runtime_error("Could not open conducting atmosphere solution!");
        }
    }

    // Get primitives individually, so we can use GetHostMirror()
    // TODO implement VariablePack::GetHostMirror, or mirror a temporary and dump into a pack device-side
    GridScalar rho  = rc->Get("prims.rho").data; 
    GridScalar u    = rc->Get("prims.u").data; 
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P  = rc->Get("prims.B").data;
    GridScalar q;
    GridScalar dP;
    if (use_emhd) {
        q  = rc->Get("prims.q").data;
        dP = rc->Get("prims.dP").data;
    }
    // Host side mirror of primitives
    auto rho_host   = rho.GetHostMirror();
    auto u_host     = u.GetHostMirror();
    auto uvec_host  = uvec.GetHostMirror();
    auto B_host     = B_P.GetHostMirror();
    // Temporary initializations are necessary for auto type
    auto q_host     = rho.GetHostMirror();
    auto dP_host    = rho.GetHostMirror();
    if (use_emhd) {
        q_host  = q.GetHostMirror();
        dP_host = dP.GetHostMirror();
    }

    // Load coordinates 'r' and compare against grid values
    const int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    double rCoords[n1];
    double error = 0.;
    IndexRange jb_in = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    for (int i = ib.s; i <= ib.e; i++) {
        fscanf(fp_r, "%lf", &(rCoords[i]));
        GReal Xembed[GR_DIM];
        G.coord_embed(0, jb_in.s, i, Loci::center, Xembed);
        error = fabs(Xembed[1] - rCoords[i]);
        if (error > 1.e-10) {
            fprintf(stdout, "Error at radial zone i = %d, Error = %8.5e KHARMA: %8.7e, sage nb: %8.7e\n", i, error, Xembed[1], rCoords[i]);
            exit(-1);
        }
    }

    // Initialize primitives
    // TODO read->copy->assign on device?
    double rho_temp, u_temp, q_temp;

    for (int i = ib.s; i <= ib.e; i++) {
        fscanf(fp_rho, "%lf", &(rho_temp));
        fscanf(fp_u,   "%lf", &(u_temp));
        if (use_emhd)
            fscanf(fp_q, "%lf", &(q_temp));

        for (int j = jb.s; j <= jb.e; j++) {
            for (int k = kb.s; k <= kb.e; k++) {

                GReal Xnative[GR_DIM], Xembed[GR_DIM]; 
                G.coord(k, j, i, Loci::center, Xnative);
                G.coord_embed(k, j, i, Loci::center, Xembed);

                // First initialize primitives that are read from .txt files
                rho_host(k, j, i)   = rho_temp;
                u_host(k, j, i)     = u_temp;
                if (use_emhd)
                    q_host(k, j, i) = q_temp;

                // Now the remaining primitives
                uvec_host(V1, k, j, i) = 0.;
                uvec_host(V2, k, j, i) = 0.;
                uvec_host(V3, k, j, i) = 0.;
                B_host(V1, k, j, i)    = 1./(Xembed[1]*Xembed[1]*Xembed[1]);
                B_host(V2, k, j, i)    = 0.;
                B_host(V3, k, j, i)    = 0.;
                if (use_emhd)
                    dP_host(k, j, i)   = 0.;

                // Note that the velocity primitives defined up there aren't quite right.
                // For a fluid at rest wrt. the normal observer, ucon = {-1/g_tt,0,0,0}. 
                // We need to use this info to obtain the correct values for U1, U2 and U3
                // TODO is this just fourvel_to_prim?

                Real ucon[GR_DIM]         = {0};
                Real gcov[GR_DIM][GR_DIM] = {0};
                Real gcon[GR_DIM][GR_DIM] = {0};
                // Use functions because we're host-side
                G.coords.gcov_native(Xnative, gcov);
                G.coords.gcon_native(Xnative, gcon);

                ucon[0] = 1. / m::sqrt(-gcov[0][0]);
                ucon[1] = 0.;
                ucon[2] = 0.;
                ucon[3] = 0.;

                // Solve for & assign primitive velocities (utilde)
                Real u_prim[NVEC];
                fourvel_to_prim(gcon, ucon, u_prim);
                uvec_host(V1, k, j, i) = u_prim[V1];
                uvec_host(V2, k, j, i) = u_prim[V2];
                uvec_host(V3, k, j, i) = u_prim[V3];

                if (use_emhd) {
                    // Update q_host (and dP_host, which is zero in this problem). These are now q_tilde and dP_tilde
                    Real q_tilde  = q_host(k, j, i);
                    Real dP_tilde = dP_host(k, j, i);

                    if (emhd_params.higher_order_terms) {
                        Real tau, chi_e, nu_e;
                        EMHD::set_parameters_init(G, rho_temp, u_temp, emhd_params, gam, k, j, i, tau, chi_e, nu_e);
                        const Real Theta = (gam - 1.) * u_temp / rho_temp;

                        q_tilde    *= (chi_e != 0) * m::sqrt(tau / (chi_e * rho_temp * Theta * Theta));
                        dP_tilde   *= (nu_e  != 0) * m::sqrt(tau / (nu_e * rho_temp * Theta));
                    }
                    q_host(k, j, i)   = q_tilde;
                    dP_host(k, j, i)  = dP_tilde;
                }
            }
        }
    }

    // disassociate file pointer
    fclose(fp_r);
    fclose(fp_rho);
    fclose(fp_u);
    if (use_emhd)
        fclose(fp_q);

    // Deep copy to device
    Kokkos::fence();
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    if (use_emhd) {
        q.DeepCopy(q_host);
        dP.DeepCopy(dP_host);
    }
    Kokkos::fence();

    // Also fill cons.B
    B_FluxCT::BlockPtoU(rc.get(), IndexDomain::entire, false);

    Flag("Initialized");
    return TaskStatus::complete;

}
