/* Copyright 2023
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "MultiBuffer.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"


std::size_t MultiBuffer::get_metadata_size () {
    return 1 + m_nbeams;
}

std::size_t* MultiBuffer::get_metadata_location (int slice) {
    return m_metadata.dataPtr() + slice*get_metadata_size();
}

void MultiBuffer::allocate_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location == memory_location::nowhere);
    if (m_buffer_on_host) {
        m_datanodes[slice].m_buffer = reinterpret_cast<char*>(amrex::The_Pinned_Arena()->alloc(
            m_datanodes[slice].m_buffer_size * sizeof(storage_type)
        ));
        m_datanodes[slice].m_location = memory_location::pinned;
    } else {
        m_datanodes[slice].m_buffer = reinterpret_cast<char*>(amrex::The_Device_Arena()->alloc(
            m_datanodes[slice].m_buffer_size * sizeof(storage_type)
        ));
        m_datanodes[slice].m_location = memory_location::device;
    }
}

void MultiBuffer::free_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location != memory_location::nowhere);
    if (m_datanodes[slice].m_location == memory_location::pinned) {
        amrex::The_Pinned_Arena()->free(m_datanodes[slice].m_buffer);
    } else {
        amrex::The_Device_Arena()->free(m_datanodes[slice].m_buffer);
    }
    m_datanodes[slice].m_location = memory_location::nowhere;
    m_datanodes[slice].m_buffer = nullptr;
    m_datanodes[slice].m_buffer_size = 0;
}

void MultiBuffer::initialize (int nslices, int nbeams, int rank_id,
                              int n_ranks, bool buffer_on_host) {

    m_nslices = nslices;
    m_nbeams = nbeams;
    m_buffer_on_host = buffer_on_host;

    m_rank_send_to = (rank_id - 1 + n_ranks) % n_ranks;
    m_rank_receive_from = (rank_id + 1) % n_ranks;

    m_is_head_rank = rank_id + 1 == n_ranks;

    m_tag_buffer_start = 1;
    m_tag_metadata_start = m_tag_buffer_start + m_nslices;

    m_metadata.resize(get_metadata_size() * m_nslices);
    m_datanodes.resize(m_nslices);

    if (m_is_head_rank) {
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::ready_to_define;
            m_datanodes[i].m_metadata_progress = comm_progress::ready_to_define;
        }
    } else {
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::sent;
            m_datanodes[i].m_metadata_progress = comm_progress::sent;
        }
    }

    for (int i = m_nslices-1; i >= 0; --i) {
        make_progress(i, false);
    }
}


MultiBuffer::~MultiBuffer() {
    for (int i = m_nslices-1; i >= 0; --i) {
        if (m_datanodes[i].m_metadata_progress == comm_progress::send_started) {
            MPI_Cancel(&(m_datanodes[i].m_request));
        }
    }
}

void MultiBuffer::make_progress (int slice, bool is_blocking) {

    if (m_datanodes[slice].m_metadata_progress == comm_progress::ready_to_send) {
        MPI_Isend(
            get_metadata_location(slice),
            get_metadata_size(),
            amrex::ParallelDescriptor::Mpi_typemap<std::size_t>::type(),
            m_rank_send_to,
            m_tag_metadata_start + slice,
            m_comm,
            &(m_datanodes[slice].m_metadata_request));
        m_datanodes[slice].m_metadata_progress = comm_progress::send_started;
    }

    if (m_datanodes[slice].m_progress == comm_progress::ready_to_send) {
        if (m_datanodes[slice].m_buffer_size == 0) {
            m_datanodes[slice].m_progress = comm_progress::sent;
        } else {
            MPI_Isend(
                m_datanodes[slice].m_buffer,
                m_datanodes[slice].m_buffer_size,
                amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
                m_rank_send_to,
                m_tag_buffer_start + slice,
                m_comm,
                &(m_datanodes[slice].m_request));
            m_datanodes[slice].m_progress = comm_progress::send_started;
        }
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::send_started) {
        if (is_blocking) {
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::sent;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_metadata_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_metadata_progress = comm_progress::sent;
            }
        }
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::sent) {
        MPI_Irecv(
            get_metadata_location(slice),
            get_metadata_size(),
            amrex::ParallelDescriptor::Mpi_typemap<std::size_t>::type(),
            m_rank_receive_from,
            m_tag_metadata_start + slice,
            m_comm,
            &(m_datanodes[slice].m_metadata_request));
        m_datanodes[slice].m_metadata_progress = comm_progress::receive_started;
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::receive_started) {
        if (is_blocking) {
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::received;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_metadata_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_metadata_progress = comm_progress::received;
            }
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::send_started) {
        if (is_blocking) {
            MPI_Wait(&(m_datanodes[slice].m_request), MPI_STATUS_IGNORE);
            free_buffer(slice);
            m_datanodes[slice].m_progress = comm_progress::sent;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                free_buffer(slice);
                m_datanodes[slice].m_progress = comm_progress::sent;
            }
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::sent &&
        m_datanodes[slice].m_metadata_progress == comm_progress::received) {

        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location == memory_location::nowhere);

        m_datanodes[slice].m_buffer_size = get_metadata_location(slice)[0];

        if (m_datanodes[slice].m_buffer_size == 0) {
            m_datanodes[slice].m_progress = comm_progress::received;
        } else {
            allocate_buffer(slice);
            MPI_Irecv(
                m_datanodes[slice].m_buffer,
                m_datanodes[slice].m_buffer_size,
                amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
                m_rank_receive_from,
                m_tag_buffer_start + slice,
                m_comm,
                &(m_datanodes[slice].m_request));
            m_datanodes[slice].m_progress = comm_progress::receive_started;
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::receive_started) {
        if (is_blocking) {
            MPI_Wait(&(m_datanodes[slice].m_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_progress = comm_progress::received;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_progress = comm_progress::received;
            }
        }
    }

    if (is_blocking) {
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_metadata_progress == comm_progress::received);
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_progress == comm_progress::received);
    }
}

void MultiBuffer::get_data (int slice, MultiBeam& beams, int beam_slice) {
    if (m_datanodes[slice].m_progress == comm_progress::ready_to_define) {
        for (int b = 0; b < m_nbeams; ++b) {
            beams.getBeam(b).intializeSlice(slice, beam_slice);
        }
    } else {
        make_progress(slice, true);
        if (m_datanodes[slice].m_buffer_size != 0) {
            unpack_data(slice, beams, beam_slice);
            free_buffer(slice);
        }
    }
    m_datanodes[slice].m_progress = comm_progress::in_use;
    m_datanodes[slice].m_metadata_progress = comm_progress::in_use;
}

void MultiBuffer::put_data (int slice, MultiBeam& beams, int beam_slice, bool is_last_time_step) {
    if (is_last_time_step) {
        m_datanodes[slice].m_progress = comm_progress::sim_completed;
        m_datanodes[slice].m_metadata_progress = comm_progress::sim_completed;
    } else {
        write_metadata(slice, beams, beam_slice);
        if (m_datanodes[slice].m_buffer_size != 0) {
            allocate_buffer(slice);
            pack_data(slice, beams, beam_slice);
        }
        m_datanodes[slice].m_progress = comm_progress::ready_to_send;
        m_datanodes[slice].m_metadata_progress = comm_progress::ready_to_send;
    }
    for (int i = m_nslices-1; i >= 0; --i) {
        make_progress(i, false);
    }
}

void MultiBuffer::write_metadata (int slice, MultiBeam& beams, int beam_slice) {
    std::size_t offset = 0;
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = beams.getBeam(b).getNumParticles(beam_slice);
        get_metadata_location(slice)[b + 1] = num_particles;
        offset += ((num_particles + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((num_particles + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    get_metadata_location(slice)[0] = (offset+sizeof(storage_type)-1) / sizeof(storage_type);
    m_datanodes[slice].m_buffer_size = get_metadata_location(slice)[0];
    AMREX_ALWAYS_ASSERT(get_metadata_location(slice)[0] < std::numeric_limits<int>::max());
}

std::size_t MultiBuffer::get_buffer_offset_real (int slice, int ibeam, int rcomp) {
    std::size_t offset = 0;
    for (int b = 0; b < ibeam; ++b) {
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * rcomp * sizeof(amrex::Real);
    return offset;
}

std::size_t MultiBuffer::get_buffer_offset_int (int slice, int ibeam, int icomp) {
    std::size_t offset = 0;
    for (int b = 0; b < ibeam; ++b) {
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * icomp * sizeof(int);
    return offset;
}

void MultiBuffer::pack_data (int slice, MultiBeam& beams, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = beams.getBeam(b).getNumParticles(beam_slice);
        auto& soa = beams.getBeam(b).getBeamSlice(beam_slice).GetStructOfArrays();
        for (int rcomp = 0; rcomp < BeamIdx::real_nattribs_in_buffer; ++rcomp) {
#ifdef AMREX_USE_GPU
            if (m_datanodes[slice].m_location == memory_location::pinned) {
                amrex::Gpu::dtoh_memcpy_async(
                    m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                    soa.GetRealData(rcomp).dataPtr(),
                    num_particles * sizeof(amrex::Real));
            } else {
                amrex::Gpu::dtod_memcpy_async(
                    m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                    soa.GetRealData(rcomp).dataPtr(),
                    num_particles * sizeof(amrex::Real));
            }
#else
            std::memcpy(
                m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                soa.GetRealData(rcomp).dataPtr(),
                num_particles * sizeof(amrex::Real));
#endif
        }
        for (int icomp = 0; icomp < BeamIdx::int_nattribs_in_buffer; ++icomp) {
#ifdef AMREX_USE_GPU
            if (m_datanodes[slice].m_location == memory_location::pinned) {
                amrex::Gpu::dtoh_memcpy_async(
                    m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                    soa.GetIntData(icomp).dataPtr(),
                    num_particles * sizeof(int));
            } else {
                amrex::Gpu::dtod_memcpy_async(
                    m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                    soa.GetIntData(icomp).dataPtr(),
                    num_particles * sizeof(int));
            }
#else
            std::memcpy(
                m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                soa.GetIntData(icomp).dataPtr(),
                num_particles * sizeof(int));
#endif
        }
    }
    amrex::Gpu::streamSynchronize();
    for (int b = 0; b < m_nbeams; ++b) {
        beams.getBeam(b).resize(beam_slice, 0, 0);
    }
}

void MultiBuffer::unpack_data (int slice, MultiBeam& beams, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = get_metadata_location(slice)[b + 1];
        beams.getBeam(b).resize(beam_slice, num_particles, 0);
        auto& soa = beams.getBeam(b).getBeamSlice(beam_slice).GetStructOfArrays();
        for (int rcomp = 0; rcomp < BeamIdx::real_nattribs_in_buffer; ++rcomp) {
#ifdef AMREX_USE_GPU
            if (m_datanodes[slice].m_location == memory_location::pinned) {
                amrex::Gpu::htod_memcpy_async(
                    soa.GetRealData(rcomp).dataPtr(),
                    m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                    num_particles * sizeof(amrex::Real));
            } else {
                amrex::Gpu::dtod_memcpy_async(
                    soa.GetRealData(rcomp).dataPtr(),
                    m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                    num_particles * sizeof(amrex::Real));
            }
#else
            std::memcpy(
                soa.GetRealData(rcomp).dataPtr(),
                m_datanodes[slice].m_buffer + get_buffer_offset_real(slice, b, rcomp),
                num_particles * sizeof(amrex::Real));
#endif
        }
        for (int rcomp = BeamIdx::real_nattribs_in_buffer; rcomp<BeamIdx::real_nattribs; ++rcomp) {
            amrex::Real* data_ptr = soa.GetRealData(rcomp).dataPtr();
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) noexcept {
                data_ptr[i] = amrex::Real(0.);
            });
        }
        for (int icomp = 0; icomp < BeamIdx::int_nattribs_in_buffer; ++icomp) {
#ifdef AMREX_USE_GPU
            if (m_datanodes[slice].m_location == memory_location::pinned) {
                amrex::Gpu::htod_memcpy_async(
                    soa.GetIntData(icomp).dataPtr(),
                    m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                    num_particles * sizeof(int));
            } else {
                amrex::Gpu::dtod_memcpy_async(
                    soa.GetIntData(icomp).dataPtr(),
                    m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                    num_particles * sizeof(int));
            }
#else
            std::memcpy(
                soa.GetIntData(icomp).dataPtr(),
                m_datanodes[slice].m_buffer + get_buffer_offset_int(slice, b, icomp),
                num_particles * sizeof(int));
#endif
        }
        for (int icomp = BeamIdx::int_nattribs_in_buffer; icomp < BeamIdx::int_nattribs; ++icomp) {
            int* data_ptr = soa.GetIntData(icomp).dataPtr();
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) noexcept {
                data_ptr[i] = 0;
            });
        }
    }
    amrex::Gpu::streamSynchronize();
}
