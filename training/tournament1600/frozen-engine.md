# Engine snapshot before website integration

`frozen-engine.tar.gz` preserves the engine files listed in `run.json`'s
`manifest.hashes`, captured and checked before adding Light / Lightning to the
WebAssembly interface. Every archived file matches its original recorded hash.

Archive SHA-256: `6beae51db3c4ac1d7a165e671bad9d3896e0cb9770639c3f298b79473c676952`.

The tournament reports and verification records describe that frozen engine.
Website development changes the working source, so running the original
verification against the evolving checkout will correctly report a mismatch.
To reproduce the historical checks, use a separate copy of the repository,
extract this archive at its root, and run the commands in `README.md`. Do not
extract it over a checkout containing ongoing development. Models, tournament
scripts, CSV results and their original manifests remain unchanged.
