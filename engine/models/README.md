# Lightning

`lightning.cxe` is the frozen 499-trainable-parameter structured evaluator from
`training/scaling-joint/data4x-capacity500-89.cxe` (seed 89, tier 1).

- Format: `CDCEX001`, 4,252 bytes, 529 stored coefficients.
- SHA-256: `a9ba34e95d15afe1b72b2996e47a5a453bec28f24a12617c011dca0b29208308`.
- Embedded in the WebAssembly binary; no model download or runtime fallback.
- Light uses the original evaluator. Both versions use the same search,
  endgame tablebases and web MoE decision layer.

The [four-budget tournament](../../training/tournament1600/README.md) used direct
search, not the web MoE decision layer. Its results do not establish that
Lightning is stronger at every time budget or in every position.
