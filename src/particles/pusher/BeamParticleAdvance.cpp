#include "BeamParticleAdvance.H"

#include "FieldGather.H"
#include "PushBeamParticles.H"
#include "Constants.H"
#include "GetAndSetPosition.H"
#include "HipaceProfilerWrapper.H"

void
AdvanceBeamParticles (BeamParticleContainer& beam, Fields& fields,
                      amrex::Geometry const& gm, int const lev, const int islice,
                      amrex::DenseBins<BeamParticleContainer::ParticleType>& bins)
{
    HIPACE_PROFILE("AdvanceBeamParticles()");
    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
    const PhysConst phys_const = get_phys_const();

    const amrex::Real dt = Hipace::m_dt;
    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        const int depos_order_xy = Hipace::m_depos_order_xy;
        const amrex::Box tilebox = pti.tilebox().grow(
            {depos_order_xy, depos_order_xy,
             Hipace::m_depos_order_z});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields
        const amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        const amrex::MultiFab exmby(S, amrex::make_alias, FieldComps::ExmBy, 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, FieldComps::EypBx, 1);
        const amrex::MultiFab ez(S, amrex::make_alias, FieldComps::Ez, 1);
        const amrex::MultiFab bx(S, amrex::make_alias, FieldComps::Bx, 1);
        const amrex::MultiFab by(S, amrex::make_alias, FieldComps::By, 1);
        const amrex::MultiFab bz(S, amrex::make_alias, FieldComps::Bz, 1);
        // Extract FabArray for this box
        const amrex::FArrayBox& exmby_fab = exmby[pti];
        const amrex::FArrayBox& eypbx_fab = eypbx[pti];
        const amrex::FArrayBox& ez_fab = ez[pti];
        const amrex::FArrayBox& bx_fab = bx[pti];
        const amrex::FArrayBox& by_fab = by[pti];
        const amrex::FArrayBox& bz_fab = bz[pti];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

        // Extract particle properties
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        const auto uxp = soa.GetRealData(BeamIdx::ux).data();
        const auto uyp = soa.GetRealData(BeamIdx::uy).data();
        const auto uzp = soa.GetRealData(BeamIdx::uz).data();

        const auto getPosition = GetBeamParticlePosition(pti);
        const auto SetPosition = SetBeamParticlePosition(pti);
        const amrex::Real zmin = xyzmin[2];


        // Declare a DenseBins to pass it to doDepositionShapeN, although it will not be used.
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type*
            indices = nullptr;
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const * offsets = 0;
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type cell_start = 0, cell_stop = 0;
        const int slice_deposition = 1;
        if (slice_deposition){
            indices = bins.permutationPtr();
            offsets = bins.offsetsPtr();
            // The particles that are in slice islice are
            // given by the indices[cell_start:cell_stop]
            cell_start = offsets[islice];
            cell_stop  = offsets[islice+1];
        }
        int const num_particles = slice_deposition ? cell_stop-cell_start : pti.numParticles();

        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
        const amrex::Real charge_mass_ratio = - phys_const.q_e / phys_const.m_e;

        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (long idx) {
                const int ip = slice_deposition ? indices[cell_start+idx] : idx;

                amrex::ParticleReal gammap = sqrt( 1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                            + uyp[ip]*uyp[ip]*clightsq + uzp[ip]*uzp[ip]*clightsq);

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                /* first we do half a step in x,y
                 * This is not required in z, which is pushed in one step later */
                xp += dt * 0.5_rt * uxp[ip] / gammap;
                yp += dt * 0.5_rt * uyp[ip] / gammap;

                SetPosition(ip, xp, yp, zp);

                // define field at particle position reals
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN(xp, yp, zmin,
                               ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                               exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                               dx_arr, xyzmin_arr, lo, depos_order_xy, 0);

                /* use intermediate fields to calculate next (n+1) transverse
                 * momenta */
                amrex::ParticleReal ux_next = uxp[ip] + dt * charge_mass_ratio
                            * ( ExmByp + ( phys_const.c - uzp[ip] / gammap ) * Byp );
                amrex::ParticleReal uy_next = uyp[ip] + dt * charge_mass_ratio
                            * ( EypBxp + ( uzp[ip] / gammap - phys_const.c ) * Bxp );


                /* Now computing new longitudinal momentum */
                amrex::ParticleReal ux_intermediate = ( ux_next + uxp[ip] ) * 0.5_rt;
                amrex::ParticleReal uy_intermediate = ( uy_next + uyp[ip] ) * 0.5_rt;
                amrex::ParticleReal uz_intermediate = uzp[ip]
                                                      + dt * 0.5_rt * charge_mass_ratio * Ezp;

                amrex::ParticleReal gamma_intermediate = sqrt( 1.0_rt
                                        + ux_intermediate*ux_intermediate*clightsq
                                        + uy_intermediate*uy_intermediate*clightsq
                                        + uz_intermediate*uz_intermediate*clightsq );

                amrex::ParticleReal uz_next = uzp[ip] + dt * charge_mass_ratio
                          * ( Ezp + ( ux_intermediate * Byp - uy_intermediate * Bxp )
                              / gamma_intermediate );

                /* computing next gamma value */
                amrex::ParticleReal gamma_next = sqrt( 1.0_rt + uz_next*uz_next*clightsq
                                                 + ux_next*ux_next*clightsq
                                                 + uy_next*uy_next*clightsq );

                /*
                 * computing positions and setting momenta for the next timestep
                 *(n+1)
                 * The longitudinal position is updated here as well, but in
                 * first-order (i.e. without the intermediary half-step) using
                 * a simple Galilean transformation
                 */
                xp += dt * 0.5_rt * ux_next  / gamma_next;
                yp += dt * 0.5_rt * uy_next  / gamma_next;
                zp += dt * ( uz_next  / gamma_next - phys_const.c );
                SetPosition(ip, xp, yp, zp);
                uxp[ip] = ux_next;
                uyp[ip] = uy_next;
                uzp[ip] = uz_next;

          }
          );
      }
}
