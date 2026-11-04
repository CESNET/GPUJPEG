Tests on a simulated GPU
========================

GitHub's runners have no GPU, so the main workflow (`ccpp.yml`) builds GPUJPEG
but cannot run it. The workflow in `.github/workflows/simulated-gpu.yml` fills
that gap: it runs the tests on a simulated NVIDIA T4, using
[PantheonSim](https://pantheonsim.com), a GPU simulator that runs ordinary CUDA
programs on the CPU. This page explains what the job does and how to read it.

When it runs
------------

- on every pull request;
- on pushes to `devel`, `master` and `run_tests` that touch source files,
  `CMakeLists.txt` or the workflow itself.

What it does
------------

1. Installs Ubuntu's CUDA toolkit and PantheonSim with the
   [setup-pantheonsim](https://github.com/pantheongpu/setup-pantheonsim)
   action, set to simulate an `nvidia/t4`. The compiler is the real `nvcc`.
2. Installs ImageMagick and ffmpeg, which the regression tests use to compare
   images.
3. Builds GPUJPEG with:
   - `CMAKE_CUDA_ARCHITECTURES=75`, so the binaries carry PTX for the T4
     (`compute_75` also runs on every newer NVIDIA card);
   - `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`, so that at run time the simulator's
     `libcudart` can take the place of the real one.
4. Runs `ctest --test-dir build --output-on-failure`, which covers `unittests`,
   `test-colors` and `regression`. The action puts the simulator's libraries on
   `LD_LIBRARY_PATH`, so the tests start the same way they would on a machine
   with a GPU.

What it tells you
-----------------

The kernels really execute, so a green run means encoding and decoding gave the
expected results. It does not tell you anything about speed: the simulator has
no timing model, and the times in the log mean nothing. Anything that depends
on a real card (performance, driver quirks, OpenGL interop, which this
workflow does not enable) still needs a real GPU.

How long it takes
-----------------

PantheonSim is built from source the first time and then cached. A run with a
cold cache spends about 15 minutes in the first step, and the build and tests
take under 2 minutes after that. GitHub drops caches that have not been used
for a week, so the first run after a quiet spell is slow again. This is
expected.

If a run fails
--------------

Open the `Unit and regression tests on the simulated GPU` step and look at the
failing test. Usually it is a real problem in the change, as on any other CI.
Sometimes the simulator does not support a CUDA call yet: it then returns a
CUDA error and says what was missing, so a test that ignores return codes can
also misbehave. If the same code passes on a real GPU, please open an issue at
[pantheongpu/pantheonsim](https://github.com/pantheongpu/pantheonsim/issues)
with the log. These are usually fixed quickly.

Versions and safety
-------------------

The action is pinned by commit, and each release of the action pins the
simulator version it was tested with, so nothing changes in the job until the
pin is moved. To update it, replace the commit in the `uses:` line with a newer
release of `pantheongpu/setup-pantheonsim`. The workflow's token is read-only
(`permissions: contents: read`), it uses no secrets, and it installs packages
only from Ubuntu and NVIDIA.
