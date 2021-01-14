#ifndef PUSHPLASMAPARTICLES_H_
#define PUSHPLASMAPARTICLES_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "FieldGather.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"

/** \brief Pushing the plasma particles (currently only with a 5th order Adams Bashforth Pusher)
 *
 * \param[in,out] xp position in x direction
 * \param[in,out] yp position in y direction
 * \param[in,out] zp position in z direction
 * \param[in,out] uxp momentum in x direction
 * \param[in,out] uyp momentum in y direction
 * \param[in,out] psip plasma pseudo-potential
 * \param[in,out] x_prev previous position in x direction
 * \param[in,out] y_prev previous position in y direction
 * \param[in,out] ux_temp temporary momentum in x direction
 * \param[in,out] uy_temp temporary momentum in y direction
 * \param[in,out] psi_temp temporary plasma pseudo-potential
 * \param[in] Fx1 force term acting on x position of the particle
 * \param[in] Fy1 force term acting on y position of the particle
 * \param[in] Fux1 force term acting on x momentum of the particle
 * \param[in] Fuy1 force term acting on y momentum of the particle
 * \param[in] Fpsi1 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx2 force term acting on x position of the particle
 * \param[in] Fy2 force term acting on y position of the particle
 * \param[in] Fux2 force term acting on x momentum of the particle
 * \param[in] Fuy2 force term acting on y momentum of the particle
 * \param[in] Fpsi2 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx3 force term acting on x position of the particle
 * \param[in] Fy3 force term acting on y position of the particle
 * \param[in] Fux3 force term acting on x momentum of the particle
 * \param[in] Fuy3 force term acting on y momentum of the particle
 * \param[in] Fpsi3 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx4 force term acting on x position of the particle
 * \param[in] Fy4 force term acting on y position of the particle
 * \param[in] Fux4 force term acting on x momentum of the particle
 * \param[in] Fuy4 force term acting on y momentum of the particle
 * \param[in] Fpsi4 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx5 force term acting on x position of the particle
 * \param[in] Fy5 force term acting on y position of the particle
 * \param[in] Fux5 force term acting on x momentum of the particle
 * \param[in] Fuy5 force term acting on y momentum of the particle
 * \param[in] Fpsi5 force term acting on the plasma pseudo-potential of the particle
 * \param[in] dz longitudinal size of the cell
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] ip index of the plasma particle
 * \param[in] SetPosition Functor for setting the particle position
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PlasmaParticlePush (
    amrex::ParticleReal& xp, amrex::ParticleReal& yp, amrex::ParticleReal& zp,
    amrex::ParticleReal& uxp, amrex::ParticleReal& uyp, amrex::ParticleReal& psip,
    amrex::ParticleReal& x_prev, amrex::ParticleReal& y_prev, amrex::ParticleReal& ux_temp,
    amrex::ParticleReal& uy_temp, amrex::ParticleReal& psi_temp,
    const amrex::ParticleReal& Fx1,
    const amrex::ParticleReal& Fy1,
    const amrex::ParticleReal& Fux1,
    const amrex::ParticleReal& Fuy1,
    const amrex::ParticleReal& Fpsi1,
    const amrex::Real dz,
    const bool temp_slice,
    const long ip,
    const SetParticlePosition<PlasmaParticleContainer, PlasmaParticleIterator>& SetPosition
    )

{
    using namespace amrex::literals;
    /* First order adams bashforth coefficients = Euler's method has no coeffficients */

    if (!temp_slice)
    {
        xp -= dz * Fx1;
        yp -= dz * Fy1;

        SetPosition(ip, xp, yp, zp);

        x_prev = xp;
        y_prev = yp;
        uxp -= dz * Fux1;
        uyp -= dz * Fuy1;
        psip -= dz * Fpsi1;
    }
    else
    {
        xp = x_prev - ( dz * Fx1 );
        yp = y_prev - ( dz * Fy1 );

        SetPosition(ip, xp, yp, zp);

        ux_temp = uxp - ( dz * Fux1 );
        uy_temp = uyp - ( dz * Fuy1 );
        psi_temp = psip - ( dz * Fpsi1 );
    }
}

#endif //  PUSHPLASMAPARTICLES_H_
