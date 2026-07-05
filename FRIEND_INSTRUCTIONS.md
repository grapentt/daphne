# Testing the OOM fix — WSL 2 + Docker (recommended path)

Hey! Yesterday you hit the DAPHNE build OOM. I have a patch on my fork that fixes it. Would you mind trying the build with my branch and telling me whether it works on your machine?

If anything goes wrong, please copy-paste the exact commands you ran and the last ~30 lines of output back to me — that's what I need to make the fix better.

Why this route: DAPHNE's official Windows recommendation is *"Works on Ubuntu WSL2, using the provided Docker images is recommended"* ([GettingStarted.md](https://github.com/daphne-eu/daphne/blob/main/doc/GettingStarted.md#operating-system)). The dev image ships with all third-party build dependencies (LLVM, Arrow, gRPC, ANTLR, etc.) already compiled, so you skip 30-90 minutes of dep-compilation. It's also the environment I verified the fix in.

---

## What the fix does (30-second version)

The build fails because compiling DAPHNE's kernel library needs a lot of RAM per parallel job (one file needs ~3.6 GB by itself), and Ninja defaults to running as many jobs in parallel as you have CPU cores. On memory-constrained hosts, that overflows and the Linux OOM killer takes out the compiler.

My fix does three things:

1. Reworks the one really-heavy kernel so it takes ~1 GB instead of 3.6 GB to compile.
2. Adds `DAPHNE_COMPILE_JOBS` — an opt-in knob that caps **total** Ninja concurrency (all compile targets). This stops the OOM killer from hitting the non-kernel MLIR/compiler files before the kernel phase is even reached.
3. Adds `DAPHNE_KERNEL_COMPILE_JOBS` — a finer-grained cap applied on top, only for the extra-heavy kernel translation units.

Both knobs default to unset (no throttle), so behaviour for anyone who doesn't hit the OOM is unchanged. On a constrained host you set both.

---

## Step 1: make sure WSL 2 and Docker Desktop are working

You need both, and Docker Desktop must be configured to use the WSL 2 backend.

Open **PowerShell** (not WSL) and check:

```powershell
wsl --status
```

Look for `Default Version: 2`. If it's 1, run `wsl --set-default-version 2` and re-install / re-import your distro.

Then check Docker Desktop:

```powershell
docker version
```

If that fails, install Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/), and in Settings → General enable **"Use the WSL 2 based engine"** and in Settings → Resources → WSL Integration enable your Ubuntu distro.

## Step 2: check how much memory Docker can use

Open your **WSL 2 Ubuntu shell** (`wsl` in PowerShell, or start Ubuntu from the Start menu) and run:

```bash
docker info --format '{{.MemTotal}}' | awk '{printf "%.1f GB\n", $1/1024/1024/1024}'
```

That's the memory ceiling of Docker's underlying VM. On Docker Desktop for Windows, this is typically 8 GB by default.

**Write that number down.** You'll pick `JOBS` based on it below.

If you want to increase it, edit `C:\Users\<you>\.wslconfig` in a text editor (create the file if it doesn't exist):

```ini
[wsl2]
memory=12GB
```

Then in PowerShell run `wsl --shutdown`, and restart Docker Desktop. Skip this step if the current allocation is fine.

## Step 3: pick the right JOBS values

| Docker VM memory | `DAPHNE_COMPILE_JOBS` | `DAPHNE_KERNEL_COMPILE_JOBS` | Wall clock |
|---|---|---|---|
| **≥ 8 GB** | `4` | `2` | ~8-10 min |
| **6 GB** | `3` | `2` | ~10-12 min |
| **4 GB** | `2` | `1` | ~15-18 min |
| **< 4 GB** | Increase in `.wslconfig` first | | |

If in doubt, start with `COMPILE_JOBS=4` / `KERNEL_COMPILE_JOBS=2`. If it still OOMs, drop both by one.

## Step 4: pull the dev image

In your WSL 2 Ubuntu shell:

```bash
docker pull daphneeu/daphne-dev:latest_X86-64_BASE
```

That's ~5 GB. One-time download. This is the image with pre-compiled third-party deps that skips 30-90 minutes of dep-compilation.

## Step 5: clone my branch

Still in the WSL 2 Ubuntu shell:

```bash
cd ~
git clone --branch fix/kernel-compile-oom https://github.com/grapentt/daphne.git daphne-oom-fix
cd daphne-oom-fix
```

The branch is called `fix/kernel-compile-oom`. Two real commits fix the OOM; a third commit is just this file.

## Step 6: run the build inside the container

**Replace the values below with your chosen values from step 3.** Copy-paste the whole block:

```bash
docker run --rm --entrypoint /bin/bash \
    --memory=8g --memory-swap=8g \
    --mount type=bind,source=$(pwd),target=/daphne \
    --mount type=tmpfs,destination=/daphne/thirdparty/installed \
    --workdir /daphne \
    -e DAPHNE_COMPILE_JOBS=4 \
    -e DAPHNE_KERNEL_COMPILE_JOBS=2 \
    daphneeu/daphne-dev:latest_X86-64_BASE \
    -c './build.sh --no-deps --installPrefix /usr/local/'
```

What each flag does:

- `--memory=8g --memory-swap=8g` — hard memory cap the container is allowed to use. Match this to your Docker VM memory (step 2). If your VM has 12 GB, use `12g`.
- `--mount type=bind,source=$(pwd),target=/daphne` — makes the cloned repo visible inside the container at `/daphne`.
- `--mount type=tmpfs,destination=/daphne/thirdparty/installed` — prevents the container's `./build.sh` from writing over the pre-baked deps at `/usr/local/`. Follows the pattern in DAPHNE's `containers/README.md`.
- `-e DAPHNE_COMPILE_JOBS=4` — caps total Ninja concurrency (all targets). This prevents OOM in the non-kernel MLIR/compiler phase.
- `-e DAPHNE_KERNEL_COMPILE_JOBS=2` — additional finer-grained cap for the extra-heavy kernel translation units.
- `--no-deps` — the image already has all third-party deps compiled at `/usr/local/`, so skip re-building them.
- `--installPrefix /usr/local/` — tells DAPHNE's build where to find the pre-baked deps.

## Step 7: what success looks like

At the end you should see something like:

```
[161/161] Linking CXX shared library lib/libAllKernels.so
...
[FINAL] Successfully built DAPHNE.
```

No `Killed signal terminated program cc1plus`. No "Killed" in the middle of the log.

Ballpark timings from my verification runs (with `DAPHNE_COMPILE_JOBS=4 DAPHNE_KERNEL_COMPILE_JOBS=2`):

- **8 GB memory**: ~8-10 min total, peak RAM around 5-6 GB.
- **4 GB memory, COMPILE_JOBS=2 KERNEL_COMPILE_JOBS=1**: ~15-18 min, peak RAM ~3.8 GB.

The compiled DAPHNE binary lands in `bin/daphne` on the host (because `/daphne` is bind-mounted).

## Step 8: what to send back

Whether it works or not, the useful things to know are:

1. **Which JOBS values you used** (`DAPHNE_COMPILE_JOBS` and `DAPHNE_KERNEL_COMPILE_JOBS`) and **your Docker VM memory** (from step 2).
2. **What `--memory=` you passed** to `docker run`.
3. **Success or failure.** If failure, the last 30 lines of output — especially anything with the words `Killed` or `cc1plus`. Paste as text, not a screenshot.
4. **Wall clock**, if you noticed it: was it in the same ballpark as step 7?
5. **Anything weird**: did the build hang, did Docker report memory-related warnings, did the image pull fail?

## Common gotchas

- **"docker: permission denied" or the daemon can't be reached.** Docker Desktop's WSL Integration isn't enabled for your distro. Go to Docker Desktop Settings → Resources → WSL Integration, toggle your Ubuntu on, restart Docker Desktop.
- **"OOM even with KERNEL_COMPILE_JOBS=1".** The non-kernel MLIR/compiler targets are OOMing before the kernel phase. Make sure you are also setting `DAPHNE_COMPILE_JOBS` (e.g. `DAPHNE_COMPILE_JOBS=2`). If it still OOMs with both set to `1`, raise `--memory=` to `6g` or bump the WSL VM in `.wslconfig`.
- **"The build gets to some kernel and then hangs forever."** More likely a Docker Desktop daemon hang than an OOM. Restart Docker Desktop from the tray icon and retry.
- **"I want to iterate on DAPHNE afterwards."** For long-term development, running DAPHNE inside Docker on top of WSL is slower per-iteration than running it directly in WSL. This doc is optimized for one-off testing. If you want to develop long-term, ask me — there's a WSL-native path but it needs `./install-ubuntu-packages.sh` and takes 30-90 min for the deps the first time.
- **"clone permission denied."** The fork is public, no auth needed. If your git is set to force SSH, either `git config --global url."https://github.com/".insteadOf "git@github.com:"` or use the HTTPS URL as shown above.

Thanks a ton for testing!
