.. _build-source:

Build/install Hipace++
======================

Developers
----------

If you are new to CMake, `this short tutorial <https://hsf-training.github.io/hsf-training-cmake-webpage/>`__ from the HEP Software foundation is the perfect place to get started with it. If you just want to use CMake to build the project, jump into sections *1. Introduction*, *2. Building with CMake* and *9. Finding Packages*.

Dependencies
------------

Hipace++ depends on the following popular third party software.
Please see installation instructions below in the Developers section.

- a mature `C++14 <https://en.wikipedia.org/wiki/C%2B%2B14>`__ compiler: e.g. GCC 5, Clang 3.6 or newer
- `CMake 3.15.0+ <https://cmake.org/>`__
- `AMReX development <https://amrex-codes.github.io>`__: we automatically download and compile a copy of AMReX
- `openPMD-api dev <https://github.com/openPMD/openPMD-api>`__: we automatically download and compile a copy of openPMD-api

  - `HDF5 <https://support.hdfgroup.org/HDF5>`__ 1.8.13+ (optional; for ``.h5`` file support)
  - `ADIOS2 <https://github.com/ornladios/ADIOS2>`__ 2.6.0+ (optional; for ``.bp`` file support)
- Nvidia GPU support: `CUDA Toolkit 9.0+ <https://developer.nvidia.com/cuda-downloads>`__ (see `matching host-compilers <https://gist.github.com/ax3l/9489132>`__)
- CPU-only: `FFTW3 <http://www.fftw.org/>`__ (only used serially; *not* needed for Nvidia GPUs)

Optional dependencies include:

- `MPI 3.0+ <https://www.mpi-forum.org/docs/>`__: for multi-node and/or multi-GPU execution
- `OpenMP 3.1+ <https://www.openmp.org>`__: for threaded CPU execution (currently not fully accelerated)
- `CCache <https://ccache.dev>`__: to speed up rebuilds (needs 3.7.9+ for CUDA)

Install Dependencies
--------------------

macOS/Linux:

.. code-block:: bash

   spack env create hipace-dev
   spack env activate hipace-dev
   spack add adios2  # for .bp file support
   spack add ccache
   spack add cmake
   spack add fftw
   spack add hdf5    # for .h5 file support
   spack add mpi
   spack add pkgconfig  # for fftw
   # optional:
   # spack add cuda
   spack install

(in new terminals, re-activate the environment with ``spack env activate hipace-dev`` again)

or macOS/Linux:

.. code-block:: bash

   brew update
   brew install adios2  # for .bp file support
   brew install ccache
   brew install cmake
   brew install fftw
   brew install hdf5-mpi  # for .h5 file support
   brew install libomp
   brew install pkg-config  # for fftw
   brew install open-mpi

Now, ``cmake --version`` should be at version 3.15.0 or newer.

Configure your compiler
-----------------------

For example, using a GCC on macOS:

.. code-block:: bash

   export CC=$(which gcc)
   export CXX=$(which g++)


If you also want to select a CUDA compiler:

.. code-block:: bash

   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=$(which g++)


Build & Test
------------

From the base of the Hipace++ source directory, execute:

.. code-block:: bash

   # find dependencies & configure
   cmake -S . -B build

   # build using up to four threads
   cmake --build build -j 4

   # run tests
   (cd build; ctest --output-on-failure)


An executable Hipace++ binary with the current compile-time options encoded in its file name will be created in ``bin/``.
Additionally, a `symbolic link <https://en.wikipedia.org/wiki/Symbolic_link>`__ named ``hipace`` can be found in that directory, which points to the last built Hipace++ executable. You can inspect and modify build options after running `cmake ..` with either

.. code-block:: bash

   ccmake build

or by providing arguments to the CMake call

.. code-block:: bash

   cmake -S . -B build -D<OPTION_A>=<VALUE_A> -D<OPTION_B>=<VALUE_B>


=============================  ========================================  =====================================================
 CMake Option                  Default & Values                          Description
-----------------------------  ----------------------------------------  -----------------------------------------------------
 ``CMAKE_BUILD_TYPE``          **RelWithDebInfo**/Release/Debug          Type of build, symbols & optimizations
 ``HiPACE_COMPUTE``            **NOACC**/CUDA/SYCL/HIP/OMP               On-node, accelerated computing backend
 ``HiPACE_MPI``                **ON**/OFF                                Multi-node support (message-passing)
 ``HiPACE_PRECISION``          SINGLE/**DOUBLE**                         Floating point precision (single/double)
 ``HiPACE_amrex_repo``         https://github.com/AMReX-Codes/amrex.git  Repository URI to pull and build AMReX from
 ``HiPACE_amrex_branch``       ``development``                           Repository branch for ``HiPACE_amrex_repo``
 ``HiPACE_amrex_internal``     **ON**/OFF                                Needs a pre-installed AMReX library if set to ``OFF``
 ``HiPACE_OPENPMD``            **ON**/OFF                                openPMD I/O (HDF5, ADIOS2)
=============================  ========================================  =====================================================

Hipace++ can be configured in further detail with options from AMReX, which are `documented in the AMReX manual <https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options>`.

**Developers** might be interested in additional options that control dependencies of Hipace++.
By default, the most important dependencies of Hipace++ are automatically downloaded for convenience:

===========================  =============================================  ==========================================================
CMake Option                 Default & Values                               Description
---------------------------  ---------------------------------------------  ----------------------------------------------------------
``HiPACE_amrex_src``         *None*                                         Path to AMReX source directory (preferred if set)
``HiPACE_amrex_repo``        `https://github.com/AMReX-Codes/amrex.git`     Repository URI to pull and build AMReX from
``HiPACE_amrex_branch``      ``development``                                Repository branch for ``HiPACE_amrex_repo``
``HiPACE_amrex_internal``    **ON**/OFF                                     Needs a pre-installed AMReX library if set to ``OFF``
``HiPACE_openpmd_src``       *None*                                         Path to openPMD-api source directory (preferred if set)
``HiPACE_openpmd_repo``      `https://github.com/openPMD/openPMD-api.git`   Repository URI to pull and build openPMD-api from
``HiPACE_openpmd_branch``    `0.13.2`                                       Repository branch for ``HiPACE_openpmd_repo``
``HiPACE_openpmd_internal``  **ON**/OFF                                     Needs a pre-installed openPMD-api library if set to ``OFF``
===========================  =============================================  ==========================================================

For example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in ``$HOME/src/amrex``, add the ``cmake`` argument ``-DHiPACE_amrex_src=$HOME/src/amrex``.
Relative paths are also supported, e.g. ``-DHiPACE_amrex_src=../amrex``.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to ``https://github.com/WeiqunZhang/amrex/`` in a branch ``new-feature`` then pass to ``cmake`` the arguments: ``-DHiPACE_amrex_repo=https://github.com/WeiqunZhang/amrex.git -DHiPACE_amrex_branch=new-feature``.

You can speed up the install further if you pre-install these dependencies, e.g. with a package manager.
Set ``-DHiPACE_<dependency-name>_internal=OFF`` and add installation prefix of the dependency to the environment variable `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html>`.
Please see the [short CMake tutorial that we linked above](#Developers) if this sounds new to you.

Run a first simulation and look at the results
----------------------------------------------

After compiling Hipace++ (see above), from the Hipace++ root directory, execute

.. code-block:: bash

   cd examples/linear_wake/
   ../../build/bin/hipace inputs # Run the simulation with fields and plasma and beam particles
   ./analysis.py # Plot results with the [openPMD-viewer](https://github.com/openPMD/openPMD-viewer), and save in img.png


Documentation
-------------

The documentation is written at the `RST <https://sphinx-tutorial.readthedocs.io/step-1/>`__ format, to compile the documentation locally use

.. code-block::bash

   cd docs
   pip install -r requirements.txt # only the first time
   make html
   open build/html/index.html

The last line would work on MacOS. On another platform, open the html file with your favorite browser.

Style and conventions
---------------------

- All new element (class, member of a class, struct, function) declared in a .H file must have a Doxygen-readable documentation
- Indent four spaces
- No tabs allowed
- No end-of-line whitespaces allowed
- Classes use CamelCase
- Objects use snake_case
- Lines should not have >100 characters
- The declaration and definition of a function should have a space between the function name and the first bracket (``my_function (...)``), function calls should not (``my_function(...)``).
  This is a convention introduce in AMReX so ``git grep "my_function ("`` returns only the declaration and definition, not the many function calls.
