/* 
 *  File: orszag_tang.hpp
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
#pragma once

#include "decs.hpp"
#include "eos.hpp"

using namespace std;
using namespace parthenon;

/**
 * relativistic version of the Orszag-Tang vortex 
 * Orszag & Tang 1979, JFM 90, 129-143. 
 * original OT problem was incompressible 
 * this is based on compressible version given
 * in Toth 2000, JCP 161, 605.
 * 
 * in the limit tscale -> 0 the problem is identical
 * to the nonrelativistic problem; as tscale increases
 * the problem becomes increasingly relativistic
 * 
 * Stolen directly from iharm2d_v3
 */
void InitializeOrszagTang(MeshBlock *pmb, const GRCoordinates& G, const GridVars& P, Real tscale=0.05)
{
    // Puts the current sheet in the middle of the domain
    Real phase = M_PI;

    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            P(prims::rho, k, j, i) = 25./9.;
            P(prims::u, k, j, i) = 5./(3.*(gam - 1.));
            P(prims::u1, k, j, i) = -sin(X[2] + phase);
            P(prims::u2, k, j, i) = sin(X[1] + phase);
            P(prims::u3, k, j, i) = 0.;
            P(prims::B1, k, j, i) = -sin(X[2] + phase);
            P(prims::B2, k, j, i) = sin(2.*(X[1] + phase));
            P(prims::B3, k, j, i) = 0.;
        }
    );
    // Rescale primitive velocities & B field by tscale, and internal energy by the square.
    pmb->par_for("ot_renorm", prims::u, NPRIM-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VARS {
            P(p, k, j, i) *= tscale * (p == prims::u ? tscale : 1);
        }
    );
}
