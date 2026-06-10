# n_pendulum

Minimal PufferLib `n_pendulum` environment export.

This repo intentionally contains only files related to the custom environment:

```text
config/n_pendulum.ini
ocean/n_pendulum/binding.c
ocean/n_pendulum/n_pendulum.c
ocean/n_pendulum/n_pendulum.h
ocean/n_pendulum/changelog.md
```

## Install Into PufferLib

From this repo:

```bash
cp config/n_pendulum.ini /path/to/pufferlib/config/n_pendulum.ini
mkdir -p /path/to/pufferlib/ocean/n_pendulum
cp ocean/n_pendulum/* /path/to/pufferlib/ocean/n_pendulum/
```

Then from the PufferLib checkout:

```bash
./build.sh n_pendulum
puffer train n_pendulum
```

Use `puffer eval n_pendulum --load-model-path latest` to render the latest
checkpoint after training.

## Notes

- Current target is fixed 5 links with `DP_MAX_LINKS = 7`.
- Link-count curriculum has been removed.
- Current policy is a `100 x 4` MinGRU, about `126k` params.
- See `ocean/n_pendulum/changelog.md` for reward, model, and performance notes.
