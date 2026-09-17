# Benchmarks

This directory contains the benchmark problems used in this repository.

Each benchmark includes two simulation workflows:

- **initial_run** – runs the simulation from the prescribed initial condition at time *t = 0*.
- **restart_run** – runs the simulation from a user-defined state, providing the restart workflow used during data assimilation (DA).

The two workflows are organized consistently across all benchmark problems:

```text
benchmarks/
├── sod/
│   ├── initial_run/
│   └── restart_run/
│
├── toro/
│   ├── initial_run/
│   └── restart_run/
│
└── blast_wave/
    ├── initial_run/
    └── restart_run/
```

See the README file within each benchmark directory for case-specific setup and execution instructions.
