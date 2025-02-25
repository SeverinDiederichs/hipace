#ifndef HIPACE_BinSort_H_
#define HIPACE_BinSort_H_

#include "BeamParticleContainer.H"
#include "BoxSort.H"

#include <AMReX_MultiFab.H>

using BeamBins = amrex::DenseBins<BeamParticleContainer::ParticleType>;

/** \brief Find particles that are in each slice, and return collections of indices per slice.
 *
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] lev MR level
 * \param[in] ibox index of the box
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] beam Beam particle container
 * \param[in] geom Geometry
 * \param[in] a_box_sorter object that sorts particles by box
 */
BeamBins
findParticlesInEachSlice (
    int lev, int ibox, amrex::Box bx,
    BeamParticleContainer& beam, const amrex::Geometry& geom,
    const BoxSorter& a_box_sorter);

#endif // HIPACE_BinSort_H_
