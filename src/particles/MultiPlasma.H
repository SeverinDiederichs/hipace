#ifndef MULTIPLASMA_H_
#define MULTIPLASMA_H_

#include "PlasmaParticleContainer.H"
#include "fields/Fields.H"

class MultiPlasma
{

public:

    /** Constructor
     * \param[in] amr_core AmrCore object
     */
    MultiPlasma (amrex::AmrCore* amr_core);

    /** Destructor */
    ~MultiPlasma () {}

    /** \brief Loop over plasma species and initialize them.
     *
     * \param[in] lev MR level
     * \param[in] slice_ba slice boxarray, on which plasma particles are defined
     * \param[in] slice_dm DistributionMapping of the transverse slice domain
     * \param[in] slice_gm slice geometry
     * \param[in] gm Geometry of the simulation, to get the cell size
     */
    void InitData (int lev, amrex::BoxArray slice_ba, amrex::DistributionMapping slice_dm,
                   amrex::Geometry slice_gm, amrex::Geometry gm);


    /** Loop over plasma species and depose their currents into the current 2D slice in fields
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] which_slice defines if this or the next slice is handled
     * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
     * \param[in] deposit_jx_jy if true, deposit to jx and jy
     * \param[in] deposit_jz if true, deposit to jz
     * \param[in] deposit_rho if true, deposit to rho
     * \param[in] deposit_j_squared if true, deposit jxx, jxy, jyy
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void DepositCurrent (
        Fields & fields, int which_slice, bool temp_slice, bool deposit_jx_jy, bool deposit_jz,
        bool deposit_rho, bool deposit_j_squared, amrex::Geometry const& gm, int const lev);

    /** \brief Return max density, to compute the adaptive time step.
     *
     * the max is taken across species AND include m_adaptive_density, giving a way to
     * specify a density to the adaptive time step calculator even with no plasma species.
     */
    amrex::Real maxDensity ();

    /** \brief Loop over plasma species and Gather fields, update forces and push particles
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
     * \param[in] do_push boolean to define if plasma particles are pushed
     * \param[in] do_update boolean to define if the force terms are updated
     * \param[in] do_shift boolean to define if the force terms are shifted
     * \param[in] lev MR level
     */
    void AdvanceParticles (
        Fields & fields, amrex::Geometry const& gm, bool temp_slice, bool do_push,
        bool do_update, bool do_shift, int lev);

    /** \brief Resets the particle position x, y, to x_prev, y_prev
     *
     * \param[in] lev MR level
     * \param[in] initial whether this is initial reset at each time step. iF so, reset everything.
     */
    void ResetParticles (int lev, bool initial=false);

    /** \brief Loop over plasma species and deposit their neutralizing background, if needed
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] which_slice slice in which the densities are deposited
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void DepositNeutralizingBackground (
        Fields & fields, int which_slice, amrex::Geometry const& gm, int const lev);

    /** Calculates Ionization Probability and makes new Plasma Particles
     *
     * \param[in] lev MR level
     * \param[in] geom Geometry of the simulation, to get the cell size
     * \param[in] fields the general field class
     */
    void DoFieldIonization (const int lev, const amrex::Geometry& geom, Fields& fields);

    /** \brief whether all plasma species use a neutralizing background, e.g. no ion motion */
    bool AllSpeciesNeutralizeBackground () const;
private:

    amrex::Vector<PlasmaParticleContainer> m_all_plasmas; /**< contains all plasma containers */
    amrex::Vector<std::string> m_names; /**< names of all plasma containers */
    int m_nplasmas; /**< number of plasma containers */
    /** Background (hypothetical) density, used to compute the adaptive time step */
    amrex::Real m_adaptive_density;
};

#endif // MULTIPLASMA_H_
