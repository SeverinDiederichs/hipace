#include "MultiBeam.H"
#include "deposition/BeamDepositCurrent.H"
#include "particles/BinSort.H"
#include "pusher/BeamParticleAdvance.H"
#include "utils/HipaceProfilerWrapper.H"

MultiBeam::MultiBeam (amrex::AmrCore* /*amr_core*/)
{

    amrex::ParmParse pp("beams");
    pp.getarr("names", m_names);
    if (m_names[0] == "no_beam") return;
    m_nbeams = m_names.size();
    MultiFromFileMacro(m_names);
    for (int i = 0; i < m_nbeams; ++i) {
        m_all_beams.emplace_back(BeamParticleContainer(m_names[i]));
    }
    m_n_real_particles.resize(m_nbeams, 0);
}

void
MultiBeam::InitData (const amrex::Geometry& geom)
{
    for (auto& beam : m_all_beams) {
        beam.InitData(geom);
    }
}

void
MultiBeam::DepositCurrentSlice (
    Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, int islice,
    const amrex::Box bx, amrex::Vector<BeamBins> bins,
    const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox,
    const bool do_beam_jx_jy_deposition, const int which_slice)

{
    for (int i=0; i<m_nbeams; i++) {
        const int nghost = m_all_beams[i].numParticles() - m_n_real_particles[i];
        ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev, islice, bx,
                              a_box_sorter_vec[i].boxOffsetsPtr()[ibox], bins[i],
                              do_beam_jx_jy_deposition, which_slice, nghost);
    }
}

amrex::Vector<amrex::Vector<BeamBins>>
MultiBeam::findParticlesInEachSlice (int nlev, int ibox, amrex::Box bx,
                                     amrex::Vector<amrex::Geometry> const& geom,
                                     const amrex::Vector<BoxSorter>& a_box_sorter_vec)
{
    amrex::Vector<amrex::Vector<BeamBins>> bins;
    for (int lev = 0; lev < nlev; ++lev) {
        amrex::Vector<BeamBins> bins_per_level;
        for (int i=0; i<m_nbeams; i++) {
            bins_per_level.emplace_back(::findParticlesInEachSlice(lev, ibox, bx, m_all_beams[i],
                                                                   geom, a_box_sorter_vec[i]));
        }
        bins.emplace_back(bins_per_level);
    }
    return bins;
}

void
MultiBeam::sortParticlesByBox (
            amrex::Vector<BoxSorter>& a_box_sorter_vec,
            const amrex::BoxArray a_ba, const amrex::Geometry& a_geom)
{
    a_box_sorter_vec.resize(m_nbeams);
    for (int i=0; i<m_nbeams; i++) {
        a_box_sorter_vec[i].sortParticlesByBox(m_all_beams[i], a_ba, a_geom);
    }
}

void
MultiBeam::AdvanceBeamParticlesSlice (
    Fields& fields, amrex::Geometry const& gm, int const lev, const int islice, const amrex::Box bx,
    amrex::Vector<BeamBins>& bins,
    const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox)
{
    for (int i=0; i<m_nbeams; i++) {
        ::AdvanceBeamParticlesSlice(m_all_beams[i], fields, gm, lev, islice, bx,
                                    a_box_sorter_vec[i].boxOffsetsPtr()[ibox], bins[i]);
    }
}

int
MultiBeam::NumRealComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumRealComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumRealComps() == comps);
        }
    }
    return comps;
}

int
MultiBeam::NumIntComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumIntComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumIntComps() == comps);
        }
    }
    return comps;
}

void
MultiBeam::StoreNRealParticles ()
{
    for (int i=0; i<m_nbeams; i++) {
        m_n_real_particles[i] = m_all_beams[i].numParticles();
    }
}

int
MultiBeam::NGhostParticles (int ibeam, amrex::Vector<BeamBins>& bins, amrex::Box bx)
{
    BeamBins::index_type const * offsets = 0;
    offsets = bins[ibeam].offsetsPtr();
    return offsets[bx.bigEnd(Direction::z)+1-bx.smallEnd(Direction::z)]
        - offsets[bx.bigEnd(Direction::z)-bx.smallEnd(Direction::z)];
}

void
MultiBeam::RemoveGhosts ()
{
    for (int i=0; i<m_nbeams; i++){
        m_all_beams[i].resize(m_n_real_particles[i]);
    }
}

void
MultiBeam::PackLocalGhostParticles (int it, const amrex::Vector<BoxSorter>& box_sorters)
{
    HIPACE_PROFILE("MultiBeam::PackLocalGhostParticles()");
    for (int ibeam=0; ibeam<m_nbeams; ibeam++){

        const int offset_box_left = box_sorters[ibeam].boxOffsetsPtr()[it];
        const int offset_box_curr = box_sorters[ibeam].boxOffsetsPtr()[it+1];
        const int nghost = offset_box_curr - offset_box_left;

        // Resize particle array
        auto& ptile = getBeam(ibeam);
        int old_size = ptile.numParticles();
        auto new_size = old_size + nghost;
        ptile.resize(new_size);

        // Copy particles in box it to ghost particles
        // Access AoS particle data
        auto& aos = ptile.GetArrayOfStructs();
        const auto& pos_structs_src = aos.begin() + offset_box_left;
        const auto& pos_structs_dst = aos.begin() + old_size;
        // Access SoA particle data
        auto& soa = ptile.GetStructOfArrays(); // For momenta and weights
        const auto  wp_src = soa.GetRealData(BeamIdx::w).data()  + offset_box_left;
        const auto uxp_src = soa.GetRealData(BeamIdx::ux).data() + offset_box_left;
        const auto uyp_src = soa.GetRealData(BeamIdx::uy).data() + offset_box_left;
        const auto uzp_src = soa.GetRealData(BeamIdx::uz).data() + offset_box_left;
        const auto  wp_dst = soa.GetRealData(BeamIdx::w).data()  + old_size;
        const auto uxp_dst = soa.GetRealData(BeamIdx::ux).data() + old_size;
        const auto uyp_dst = soa.GetRealData(BeamIdx::uy).data() + old_size;
        const auto uzp_dst = soa.GetRealData(BeamIdx::uz).data() + old_size;

        amrex::ParallelFor(
            nghost,
            [=] AMREX_GPU_DEVICE (long idx) {
                pos_structs_dst[idx].id() = pos_structs_src[idx].id();
                pos_structs_dst[idx].pos(0) = pos_structs_src[idx].pos(0);
                pos_structs_dst[idx].pos(1) = pos_structs_src[idx].pos(1);
                pos_structs_dst[idx].pos(2) = pos_structs_src[idx].pos(2);
                wp_dst[idx] = wp_src[idx];
                uxp_dst[idx] = uxp_src[idx];
                uyp_dst[idx] = uyp_src[idx];
                uzp_dst[idx] = uzp_src[idx];
            }
            );
    }
}

void
MultiBeam::MultiFromFileMacro (const amrex::Vector<std::string> beam_names)
{
    amrex::ParmParse pp("beams");
    std::string all_input_file = "";
    if(!pp.query("all_from_file", all_input_file)) {
        return;
    }

    int iteration = 0;
    const bool multi_iteration = pp.query("iteration", iteration);
    amrex::Real plasma_density = 0;
    const bool multi_plasma_density = pp.query("plasma_density", plasma_density);
    std::vector<std::string> file_coordinates_xyz;
    const bool multi_file_coordinates_xyz = pp.queryarr("file_coordinates_xyz",
                                                        file_coordinates_xyz);

    for( std::string name : beam_names ) {
        amrex::ParmParse pp_beam(name);
        if(!pp_beam.contains("injection_type")) {
            std::string str_from_file = "from_file";
            pp_beam.add("injection_type", str_from_file);
            pp_beam.add("input_file", all_input_file);

            if(!pp_beam.contains("iteration") && multi_iteration) {
                pp_beam.add("iteration", iteration);
            }

            if(!pp_beam.contains("plasma_density") && multi_plasma_density) {
                pp_beam.add("plasma_density", plasma_density);
            }

            if(!pp_beam.contains("file_coordinates_xyz") && multi_file_coordinates_xyz) {
                pp_beam.addarr("file_coordinates_xyz", file_coordinates_xyz);
            }
        }
    }
}
