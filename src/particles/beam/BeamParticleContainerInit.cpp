/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Severin Diederichs, Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "particles/particles_utils/ParticleUtil.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include <AMReX_REAL.H>

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#include <iostream> // std::cout
#include <memory>   // std::shared_ptr
#endif  // HIPACE_USE_OPENPMD



namespace
{
    /** \brief Adds a single beam particle
     *
     * \param[in,out] rarrdata real array with SoA beam data
     * \param[in,out] iarrdata int array with SoA beam data
     * \param[in] x position in x
     * \param[in] y position in y
     * \param[in] z position in z
     * \param[in] ux gamma * beta_x
     * \param[in] uy gamma * beta_y
     * \param[in] uz gamma * beta_z
     * \param[in] weight weight of the single particle
     * \param[in] pid particle ID to be assigned to the particle
     * \param[in] ip index of the particle
     * \param[in] speed_of_light speed of light in the current units
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AddOneBeamParticle (
        amrex::GpuArray<amrex::ParticleReal*, BeamIdx::real_nattribs_in_buffer> rarrdata,
        amrex::GpuArray<int*, BeamIdx::int_nattribs_in_buffer> iarrdata, const amrex::Real& x,
        const amrex::Real& y, const amrex::Real& z, const amrex::Real& ux, const amrex::Real& uy,
        const amrex::Real& uz, const amrex::Real& weight, const int& pid,
        const int& ip, const amrex::Real& speed_of_light) noexcept
    {
        rarrdata[BeamIdx::x  ][ip] = x;
        rarrdata[BeamIdx::y  ][ip] = y;
        rarrdata[BeamIdx::z  ][ip] = z;
        rarrdata[BeamIdx::ux ][ip] = ux * speed_of_light;
        rarrdata[BeamIdx::uy ][ip] = uy * speed_of_light;
        rarrdata[BeamIdx::uz ][ip] = uz * speed_of_light;
        rarrdata[BeamIdx::w  ][ip] = std::abs(weight);

        iarrdata[BeamIdx::id ][ip] = pid > 0 ? pid + ip : pid;
    }
}

void
BeamParticleContainer::
InitBeamFixedPPC (const amrex::IntVect& a_num_particles_per_cell,
                  const GetInitialDensity& get_density,
                  const GetInitialMomentum& get_momentum,
                  const amrex::Geometry& a_geom,
                  const amrex::Real a_zmin,
                  const amrex::Real a_zmax,
                  const amrex::Real a_radius,
                  const amrex::RealVect a_position_mean,
                  const amrex::Real a_min_density,
                  const amrex::Vector<int>& random_ppc)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");

    if (!Hipace::HeadRank()) { return; }
 
    const amrex::IntVect ncells = a_geom.Domain().length();
    amrex::Long ncells_total = (amrex::Long) ncells[0] * ncells[1] * ncells[2];
    if ( ncells_total / Hipace::m_beam_injection_cr / Hipace::m_beam_injection_cr
         > std::numeric_limits<int>::max() / 100 ){
        amrex::Print()<<"WARNING: the number of cells is close to overflowing the maximum int,\n";
        amrex::Print()<<"consider using a larger hipace.beam_injection_cr\n";
    }

    // Since each box is allows to be very large, its number of cells may exceed the largest
    // int (~2.e9). To avoid this, we use a coarsened box (the coarsening ratio is cr, see below)
    // to inject particles. This is just a trick to have fewer cells, it injects the same
    // by using fewer larger cells and more particles per cell.
    amrex::IntVect cr {Hipace::m_beam_injection_cr,Hipace::m_beam_injection_cr,1};
    AMREX_ALWAYS_ASSERT(cr[AMREX_SPACEDIM-1] == 1);
    auto dx = a_geom.CellSizeArray();
    for (int i=0; i<AMREX_SPACEDIM; i++) dx[i] *= cr[i];
    const auto plo = a_geom.ProbLoArray();

    amrex::IntVect ppc_cr = a_num_particles_per_cell;
    for (int i=0; i<AMREX_SPACEDIM; i++) ppc_cr[i] *= cr[i];

    const int num_ppc = AMREX_D_TERM( ppc_cr[0], *ppc_cr[1], *ppc_cr[2]);

    const amrex::Real scale_fac = Hipace::m_normalized_units ?
        1./num_ppc*cr[0]*cr[1]*cr[2] : dx[0]*dx[1]*dx[2]/num_ppc;

    const amrex::Real x_mean = a_position_mean[0];
    const amrex::Real y_mean = a_position_mean[1];

    // First: loop over all cells, and count the particles effectively injected.
    amrex::Box domain_box = a_geom.Domain();
    domain_box.coarsen(cr);
    const auto lo = amrex::lbound(domain_box);
    const auto hi = amrex::ubound(domain_box);

    amrex::Gpu::DeviceVector<unsigned int> counts(domain_box.numPts(), 0);
    unsigned int* pcount = counts.dataPtr();

    amrex::Gpu::DeviceVector<unsigned int> offsets(domain_box.numPts());
    unsigned int* poffset = offsets.dataPtr();

    const amrex::GpuArray<int, 3> rand_ppc {random_ppc[0], random_ppc[1], random_ppc[2]};

    amrex::ParallelForRNG(
        domain_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, const amrex::RandomEngine& engine) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, ppc_cr, i_part, engine, rand_ppc);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                if (rand_ppc[0] + rand_ppc[1] + rand_ppc[2] == false ) {
                    // If particles are evenly spaced, discard particles
                    // individually if they are out of bounds
                    if (z >= a_zmax || z < a_zmin ||
                        ((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean)) > a_radius*a_radius) {
                            continue;
                        }
                } else {
                    // If particles are randomly spaced, discard particles
                    // if the cell is outside the domain
                    amrex::Real xc = plo[0]+i*dx[0];
                    amrex::Real yc = plo[1]+j*dx[1];
                    amrex::Real zc = plo[2]+k*dx[2];
                    if (zc >= a_zmax || zc < a_zmin ||
                        ((xc-x_mean)*(xc-x_mean)+(yc-y_mean)*(yc-y_mean)) > a_radius*a_radius) {
                            continue;
                        }
                }

                const amrex::Real density = get_density(x, y, z);
                if (density < a_min_density) continue;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
                unsigned int cellid = (uix * ny + uiy) * nz + uiz;
                pcount[cellid] += 1;
            }
        });

        int num_to_add = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

        // Second: allocate the memory for these particles
        auto& particle_tile = getBeamInitSlice();

        auto old_size = particle_tile.size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) return;

        amrex::GpuArray<amrex::ParticleReal*, BeamIdx::real_nattribs_in_buffer> rarrdata =
            particle_tile.GetStructOfArrays().realarray();

        amrex::GpuArray<int*, BeamIdx::int_nattribs_in_buffer> iarrdata =
            particle_tile.GetStructOfArrays().intarray();

        const int pid = BeamTileInit::ParticleType::NextID();
        BeamTileInit::ParticleType::NextID(pid + num_to_add);

        PhysConst phys_const = get_phys_const();

        amrex::ParallelForRNG(domain_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, const amrex::RandomEngine& engine) noexcept
        {
            int ix = i - lo.x;
            int iy = j - lo.y;
            int iz = k - lo.z;
            int nx = hi.x-lo.x+1;
            int ny = hi.y-lo.y+1;
            int nz = hi.z-lo.z+1;
            unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
            unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
            unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;

            int pidx = int(poffset[cellid] - poffset[0]);

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, ppc_cr, i_part, engine, rand_ppc);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                if (rand_ppc[0] + rand_ppc[1] + rand_ppc[2] == false) {
                    // If particles are evenly spaced, discard particles
                    // individually if they are out of bounds
                    if (z >= a_zmax || z < a_zmin ||
                        ((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean)) > a_radius*a_radius) {
                            continue;
                        }
                } else {
                    // If particles are randomly spaced, discard particles
                    // if the cell is outside the domain
                    amrex::Real xc = plo[0]+i*dx[0];
                    amrex::Real yc = plo[1]+j*dx[1];
                    amrex::Real zc = plo[2]+k*dx[2];
                    if (zc >= a_zmax || zc < a_zmin ||
                        ((xc-x_mean)*(xc-x_mean)+(yc-y_mean)*(yc-y_mean)) > a_radius*a_radius) {
                            continue;
                        }
                }

                const amrex::Real density = get_density(x, y, z);
                if (density < a_min_density) continue;

                amrex::Real u[3] = {0.,0.,0.};
                get_momentum(u[0],u[1],u[2], engine);

                const amrex::Real weight = density * scale_fac;
                AddOneBeamParticle(rarrdata, iarrdata, x, y, z, u[0], u[1], u[2], weight,
                                   pid, pidx, phys_const.c);
                ++pidx;
            }
        });
}

void
BeamParticleContainer::
InitBeamFixedWeight (int num_to_add,
                     const GetInitialMomentum& get_momentum,
                     const amrex::ParserExecutor<1>& pos_mean_x,
                     const amrex::ParserExecutor<1>& pos_mean_y,
                     const amrex::Real pos_mean_z,
                     const amrex::RealVect pos_std,
                     const amrex::Real total_charge,
                     const amrex::Real z_foc,
                     const bool do_symmetrize,
                     const bool can, const amrex::Real zmin, const amrex::Real zmax)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");
    using namespace amrex::literals;

    if (num_to_add == 0) return;
    if (do_symmetrize) num_to_add /=4;

    PhysConst phys_const = get_phys_const();

    if (Hipace::HeadRank()) {

        auto& particle_tile = getBeamInitSlice();
        auto old_size = particle_tile.size();
        auto new_size = do_symmetrize? old_size + 4*num_to_add : old_size + num_to_add;
        particle_tile.resize(new_size);

        // Access particles' SoA
        amrex::GpuArray<amrex::ParticleReal*, BeamIdx::real_nattribs_in_buffer> rarrdata =
            particle_tile.GetStructOfArrays().realarray();
        amrex::GpuArray<int*, BeamIdx::int_nattribs_in_buffer> iarrdata =
            particle_tile.GetStructOfArrays().intarray();

        const int pid = BeamTileInit::ParticleType::NextID();
        BeamTileInit::ParticleType::NextID(pid + num_to_add);

        const amrex::Real duz_per_uz0_dzeta = m_duz_per_uz0_dzeta;
        const amrex::Real single_charge = m_charge;
        const amrex::Real z_mean = can ? 0.5_rt * (zmin + zmax) : pos_mean_z;

        amrex::ParallelForRNG(
            num_to_add,
            [=] AMREX_GPU_DEVICE (int i, const amrex::RandomEngine& engine) noexcept
            {
                amrex::Real x = amrex::RandomNormal(0, pos_std[0], engine);
                amrex::Real y = amrex::RandomNormal(0, pos_std[1], engine);
                const amrex::Real z = can
                    ? (amrex::Random(engine) - 0.5_rt) * (zmax - zmin)
                    : amrex::RandomNormal(0, pos_std[2], engine);
                amrex::Real u[3] = {0.,0.,0.};
                get_momentum(u[0],u[1],u[2], engine, z, duz_per_uz0_dzeta);

                // Propagate each electron ballistically for z_foc
                x -= z_foc*u[0]/get_momentum.m_u_mean[2];
                y -= z_foc*u[1]/get_momentum.m_u_mean[2];

                const amrex::Real z_central = z + z_mean;
                int valid_id = pid;
                if (z_central < zmin || z_central > zmax) valid_id = -1;

                const amrex::Real cental_x_pos = pos_mean_x(z_central);
                const amrex::Real cental_y_pos = pos_mean_y(z_central);

                amrex::Real weight = total_charge / (num_to_add * single_charge);
                if (!do_symmetrize)
                {
                    AddOneBeamParticle(rarrdata, iarrdata, cental_x_pos+x, cental_y_pos+y,
                                       z_central, u[0], u[1], u[2], weight,
                                       valid_id, i, phys_const.c);
                } else {
                    weight /= 4;
                    AddOneBeamParticle(rarrdata, iarrdata, cental_x_pos+x, cental_y_pos+y,
                                       z_central, u[0], u[1], u[2], weight,
                                       valid_id, 4*i, phys_const.c);
                    AddOneBeamParticle(rarrdata, iarrdata, cental_x_pos-x, cental_y_pos+y,
                                       z_central, -u[0], u[1], u[2], weight,
                                       valid_id, 4*i+1, phys_const.c);
                    AddOneBeamParticle(rarrdata, iarrdata, cental_x_pos+x, cental_y_pos-y,
                                       z_central, u[0], -u[1], u[2], weight,
                                       valid_id, 4*i+2, phys_const.c);
                    AddOneBeamParticle(rarrdata, iarrdata, cental_x_pos-x, cental_y_pos-y,
                                       z_central, -u[0], -u[1], u[2], weight,
                                       valid_id, 4*i+3, phys_const.c);
                }
            });
    }

    return;
}

#ifdef HIPACE_USE_OPENPMD
amrex::Real
BeamParticleContainer::
InitBeamFromFileHelper (const std::string input_file,
                        const bool coordinates_specified,
                        const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                        const amrex::Geometry& geom,
                        amrex::Real n_0,
                        const int num_iteration,
                        const std::string species_name,
                        const bool species_specified)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");

    openPMD::Datatype input_type = openPMD::Datatype::INT;
    bool species_known;
    {
        // Check what kind of Datatype is used in beam file
        auto series = openPMD::Series( input_file , openPMD::Access::READ_ONLY );

        if(!series.iterations.contains(num_iteration)) {
            amrex::Abort("Could not find iteration " + std::to_string(num_iteration) +
                                                        " in file " + input_file + "\n");
        }
        species_known = series.iterations[num_iteration].particles.contains(species_name);

        for( auto const& particle_type : series.iterations[num_iteration].particles ) {
            if( (!species_known) || (particle_type.first == species_name) ) {
                for( auto const& physical_quantity : particle_type.second ) {
                    if( physical_quantity.first != "id") {
                        for( auto const& axes_direction : physical_quantity.second ) {
                            input_type = axes_direction.second.getDatatype();
                        }
                    }
                }
            }
        }

        if( input_type == openPMD::Datatype::INT || (species_specified && !species_known)) {
            std::string err = "Error, the particle species name " + species_name +
                  " was not found or does not contain any data. The input file contains the" +
                  " following particle species names:\n";
            for( auto const& species_type : series.iterations[num_iteration].particles ) {
                err += species_type.first + "\n";
            }
            if( !species_specified ) {
                err += "Use beam.openPMD_species_name NAME to specify a paricle species\n";
            }
            amrex::Abort(err);
        }

    }

    amrex::Real ptime {0.};
    if( input_type == openPMD::Datatype::FLOAT ) {
        ptime = InitBeamFromFile<float>(input_file, coordinates_specified, file_coordinates_xyz,
                                        geom, n_0, num_iteration, species_name, species_known);
    }
    else if( input_type == openPMD::Datatype::DOUBLE ) {
        ptime = InitBeamFromFile<double>(input_file, coordinates_specified, file_coordinates_xyz,
                                         geom, n_0, num_iteration, species_name, species_known);
    }
    else{
        amrex::Abort("Unknown Datatype used in Beam Input file. Must use double or float\n");
    }
    return ptime;
}

template <typename input_type>
amrex::Real
BeamParticleContainer::
InitBeamFromFile (const std::string input_file,
                  const bool coordinates_specified,
                  const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                  const amrex::Geometry& geom,
                  amrex::Real n_0,
                  const int num_iteration,
                  const std::string species_name,
                  const bool species_specified)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");

    amrex::Real physical_time {0.};

    auto series = openPMD::Series( input_file , openPMD::Access::READ_ONLY );

    if( series.iterations[num_iteration].containsAttribute("time") ) {
        physical_time = series.iterations[num_iteration].time<input_type>();
    }

    if (!Hipace::HeadRank()) return physical_time;

    // Initialize variables to translate between names from the file and names in Hipace
    std::string name_particle ="";
    std::string name_r ="", name_rx ="", name_ry ="", name_rz ="";
    std::string name_u ="", name_ux ="", name_uy ="", name_uz ="";
    std::string name_m ="", name_mm ="";
    std::string name_q ="", name_qq ="";
    std::string name_g ="", name_gg ="";
    bool u_is_momentum = false;

    // Iterate through all matadata in file, search for unit combination for Distance, Velocity,
    // Charge, Mass. Auto detect position, weighting and coordinates if named x y z or X Y Z etc.
    for( auto const& particle_type : series.iterations[num_iteration].particles ) {
        if( (!species_specified) || (particle_type.first == species_name) ) {
            name_particle = particle_type.first;
            for( auto const& physical_quantity : particle_type.second ) {

                std::array<double,7> units = physical_quantity.second.unitDimension();

                if(units == std::array<double,7> {1., 0., 0., 0., 0., 0., 0.}) {
                    if( (!particle_type.second.contains("position")) ||
                                                      (physical_quantity.first == "position")) {
                        name_r = physical_quantity.first;
                        for( auto const& axes_direction : physical_quantity.second ) {
                            if(axes_direction.first == "x" || axes_direction.first == "X") {
                                name_rx = axes_direction.first;
                            }
                            if(axes_direction.first == "y" || axes_direction.first == "Y") {
                                name_ry = axes_direction.first;
                            }
                            if(axes_direction.first == "z" || axes_direction.first == "Z") {
                                name_rz = axes_direction.first;
                            }
                        }
                    }
                }
                else if(units == std::array<double,7> {1., 0., -1., 0., 0., 0., 0.}) {
                    // proper velocity = gamma * v
                    name_u = physical_quantity.first;
                    u_is_momentum = false;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        if(axes_direction.first == "x" || axes_direction.first == "X") {
                            name_ux = axes_direction.first;
                        }
                        if(axes_direction.first == "y" || axes_direction.first == "Y") {
                            name_uy = axes_direction.first;
                        }
                        if(axes_direction.first == "z" || axes_direction.first == "Z") {
                            name_uz = axes_direction.first;
                        }
                    }
                }
                else if(units == std::array<double,7> {1., 1., -1., 0., 0., 0., 0.}) {
                    // momentum = gamma * m * v
                    name_u = physical_quantity.first;
                    u_is_momentum = true;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        if(axes_direction.first == "x" || axes_direction.first == "X") {
                            name_ux = axes_direction.first;
                        }
                        if(axes_direction.first == "y" || axes_direction.first == "Y") {
                            name_uy = axes_direction.first;
                        }
                        if(axes_direction.first == "z" || axes_direction.first == "Z") {
                            name_uz = axes_direction.first;
                        }
                    }
                }
                else if(units == std::array<double,7> {0., 1., 0., 0., 0., 0., 0.}) {
                    name_m = physical_quantity.first;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        name_mm = axes_direction.first;
                    }
                }
                else if(units == std::array<double,7> {0., 0., 1., 1., 0., 0., 0.}) {
                    name_q = physical_quantity.first;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        name_qq = axes_direction.first;
                    }
                }
                else if(units == std::array<double,7> {0., 0., 0., 0., 0., 0., 0.}) {
                    if(physical_quantity.first == "weighting") {
                        name_g = physical_quantity.first;
                        for( auto const& axes_direction : physical_quantity.second ) {
                            name_gg = axes_direction.first;
                        }
                    }
                }
            }
        }
    }

    // Overide coordinate names with those from file_coordinates_xyz argument
    if(coordinates_specified) {
        name_rx = name_ux = file_coordinates_xyz[0];
        name_ry = name_uy = file_coordinates_xyz[1];
        name_rz = name_uz = file_coordinates_xyz[2];
    }

    // Determine whether to use momentum or normalized momentum as well as weight, charge or mass
    // set conversion factor appropriately
    const PhysConst phys_const_SI = make_constants_SI();
    const PhysConst phys_const = get_phys_const();

    std::string name_w = "", name_ww = "";
    std::string weighting_type = "";
    std::string momentum_type = "Proper velocity";

    input_type si_to_norm_pos = 1.;
    input_type si_to_norm_momentum = phys_const_SI.c;
    input_type si_to_norm_weight = 1.;

    if(u_is_momentum) {
        si_to_norm_momentum = m_mass * (phys_const_SI.m_e / phys_const.m_e) * phys_const_SI.c;
        momentum_type = "Momentum";
    }

    if(name_gg != "") {
        name_w = name_g;
        name_ww = name_gg;
        weighting_type = "Weighting";
    }
    else if(name_qq != "") {
        name_w = name_q;
        name_ww = name_qq;
        si_to_norm_weight = m_charge * (phys_const_SI.q_e / phys_const.q_e);
        weighting_type = "Charge";
    }
    else if(name_mm != "") {
        name_w = name_m;
        name_ww = name_mm;
        si_to_norm_weight = m_mass * (phys_const_SI.m_e / phys_const.m_e);
        weighting_type = "Mass";
    }
    else {
        amrex::Abort("Could not find Charge of dimension I * T in file\n");
    }

    // Abort if file necessary information couldn't be found in file
    if(name_r == "") {
        amrex::Abort("Could not find Position of dimension L in file\n");
    }
    if(name_u == "") {
        amrex::Abort("Could not find u or Momentum of dimension L / T or M * L / T in file\n");
    }
    if(name_rx == "" || name_ux == "") {
        amrex::Abort("Coud not find x coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }
    if(name_ry == "" || name_uy == "") {
        amrex::Abort("Coud not find y coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }
    if(name_rz == "" || name_uz == "") {
        amrex::Abort("Coud not find z coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }

    for(std::string name_r_c : {name_rx, name_ry, name_rz}) {
        if(!series.iterations[num_iteration].particles[name_particle][name_r].contains(name_r_c)) {
            amrex::Abort("Beam input file does not contain " + name_r_c + " coordinate in " +
            name_r + " (position)\n");
        }
    }
    for(std::string name_u_c : {name_ux, name_uy, name_uz}) {
        if(!series.iterations[num_iteration].particles[name_particle][name_u].contains(name_u_c)) {
            amrex::Abort("Beam input file does not contain " + name_u_c + " coordinate in " +
            name_u + " (momentum)\n");
        }
    }

    // print the names of the arrays in the file that are used for the simulation
    if(Hipace::m_verbose >= 3){
       amrex::Print() << "Beam Input File '" << input_file << "' in Iteration '" << num_iteration
          << "' and Paricle '" << name_particle
          << "' imported with:\nPositon '" << name_r << "' (coordinates '" << name_rx << "', '"
          << name_ry << "', '" << name_rz << "')\n"
          << momentum_type << " '" << name_u << "' (coordinates '" << name_ux
          << "', '" << name_uy << "', '" << name_uz << "')\n"
          << weighting_type << " '" << name_w << "' (in '" << name_ww << "')\n";
    }

    auto electrons = series.iterations[num_iteration].particles[name_particle];

    const auto num_to_add = electrons[name_r][name_rx].getExtent()[0];

    if(num_to_add >= 2147483647) {
        amrex::Abort("Beam can't have more than 2'147'483'646 Particles\n");
    }

    auto del = [](input_type *p){ amrex::The_Pinned_Arena()->free(reinterpret_cast<void*>(p)); };

    // copy Data to pinned memory
    std::shared_ptr<input_type> r_x_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> r_y_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> r_z_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_x_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_y_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_z_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> w_w_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};

    electrons[name_r][name_rx].loadChunk<input_type>(r_x_data, {0u}, {num_to_add});
    electrons[name_r][name_ry].loadChunk<input_type>(r_y_data, {0u}, {num_to_add});
    electrons[name_r][name_rz].loadChunk<input_type>(r_z_data, {0u}, {num_to_add});
    electrons[name_u][name_ux].loadChunk<input_type>(u_x_data, {0u}, {num_to_add});
    electrons[name_u][name_uy].loadChunk<input_type>(u_y_data, {0u}, {num_to_add});
    electrons[name_u][name_uz].loadChunk<input_type>(u_z_data, {0u}, {num_to_add});
    electrons[name_w][name_ww].loadChunk<input_type>(w_w_data, {0u}, {num_to_add});

    series.flush();

    // calculate the multiplier to convert to Hipace units
    if(Hipace::m_normalized_units) {
        if(n_0 == 0) {
            if(electrons.containsAttribute("HiPACE++_Plasma_Density")) {
                n_0 = electrons.getAttribute("HiPACE++_Plasma_Density").get<double>();
            }
            else {
                amrex::Abort("Please specify the plasma density of the external beam "
                             "to use it with normalized units with beam.plasma_density");
            }
        }
        const auto dx = geom.CellSizeArray();
        const double omega_p = (double)phys_const_SI.q_e * sqrt( (double)n_0 /
                                      ( (double)phys_const_SI.ep0 * (double)phys_const_SI.m_e ) );
        const double kp_inv = (double)phys_const_SI.c / omega_p;
        si_to_norm_pos = (input_type)kp_inv;
        si_to_norm_weight *= (input_type)( n_0 * dx[0] * dx[1] * dx[2] * kp_inv * kp_inv * kp_inv );
    }

    input_type unit_rx, unit_ry, unit_rz, unit_ux, unit_uy, unit_uz, unit_ww;
    bool hipace_restart = false;
    const std::string attr = "HiPACE++_reference_unitSI";
    if(electrons.containsAttribute("HiPACE++_use_reference_unitSI")) {
        if(electrons.getAttribute("HiPACE++_use_reference_unitSI").get<bool>() == true) {
            hipace_restart = true;
        }
    }

    if(hipace_restart) {
        unit_rx = electrons[name_r][name_rx].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_ry = electrons[name_r][name_ry].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_rz = electrons[name_r][name_rz].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_ux = electrons[name_u][name_ux].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_uy = electrons[name_u][name_uy].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_uz = electrons[name_u][name_uz].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_ww = electrons[name_w][name_ww].getAttribute(attr).get<double>() / si_to_norm_weight;
    }
    else {
        unit_rx = electrons[name_r][name_rx].unitSI() / si_to_norm_pos;
        unit_ry = electrons[name_r][name_ry].unitSI() / si_to_norm_pos;
        unit_rz = electrons[name_r][name_rz].unitSI() / si_to_norm_pos;
        unit_ux = electrons[name_u][name_ux].unitSI() / si_to_norm_momentum;
        unit_uy = electrons[name_u][name_uy].unitSI() / si_to_norm_momentum;
        unit_uz = electrons[name_u][name_uz].unitSI() / si_to_norm_momentum;
        unit_ww = electrons[name_w][name_ww].unitSI() / si_to_norm_weight;
    }

    // input data using AddOneBeamParticle function, make necessary variables and arrays
    auto& particle_tile = getBeamInitSlice();
    auto old_size = particle_tile.size();
    auto new_size = old_size + num_to_add;
    particle_tile.resize(new_size);
    amrex::GpuArray<amrex::ParticleReal*, BeamIdx::real_nattribs_in_buffer> rarrdata =
        particle_tile.GetStructOfArrays().realarray();
    amrex::GpuArray<int*, BeamIdx::int_nattribs_in_buffer> iarrdata =
        particle_tile.GetStructOfArrays().intarray();
    const int pid = BeamTileInit::ParticleType::NextID();
    BeamTileInit::ParticleType::NextID(pid + num_to_add);

    const input_type * const r_x_ptr = r_x_data.get();
    const input_type * const r_y_ptr = r_y_data.get();
    const input_type * const r_z_ptr = r_z_data.get();
    const input_type * const u_x_ptr = u_x_data.get();
    const input_type * const u_y_ptr = u_y_data.get();
    const input_type * const u_z_ptr = u_z_data.get();
    const input_type * const w_w_ptr = w_w_data.get();

    amrex::ParallelFor(int(num_to_add),
        [=] AMREX_GPU_DEVICE (const int i) {
            AddOneBeamParticle(rarrdata, iarrdata,
                static_cast<amrex::Real>(r_x_ptr[i] * unit_rx),
                static_cast<amrex::Real>(r_y_ptr[i] * unit_ry),
                static_cast<amrex::Real>(r_z_ptr[i] * unit_rz),
                static_cast<amrex::Real>(u_x_ptr[i] * unit_ux), // = gamma * beta
                static_cast<amrex::Real>(u_y_ptr[i] * unit_uy),
                static_cast<amrex::Real>(u_z_ptr[i] * unit_uz),
                static_cast<amrex::Real>(w_w_ptr[i] * unit_ww),
                pid, i, phys_const.c);
        });

    amrex::Gpu::streamSynchronize();

    return physical_time;
}
#endif // HIPACE_USE_OPENPMD
