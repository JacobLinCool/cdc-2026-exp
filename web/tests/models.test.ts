import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { CdcGame, EngineModel, initSync } from "../src/pkg/cdc_engine.js";
import { decodeRecord, encodeRecord, type GameRecord } from "../src/replay.ts";

initSync({ module: readFileSync(new URL("../src/pkg/cdc_engine_bg.wasm", import.meta.url)) });

test("Lightning is the exact frozen 499-parameter checkpoint", () => {
  const bytes = readFileSync(new URL("../../engine/models/lightning.cxe", import.meta.url));
  assert.equal(bytes.length, 4252);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), "a9ba34e95d15afe1b72b2996e47a5a453bec28f24a12617c011dca0b29208308");
});

test("WASM keeps both slots' models through searching and undo", () => {
  for (const a of [EngineModel.Light, EngineModel.Lightning]) {
    for (const b of [EngineModel.Light, EngineModel.Lightning]) {
      const game = new CdcGame(731, a, b);
      try {
        for (let square = 0; square < 24; square++) assert.notEqual(game.play(square, square), 255);
        for (const slot of [0, 1]) {
          game.set_variety(slot, 0);
          const analysis = JSON.parse(game.think(slot, 10));
          assert.equal(analysis.model, [a, b][slot] === EngineModel.Light ? "light" : "lightning");
          assert.ok(game.legal_moves().includes(analysis.move.from | (analysis.move.to << 5)));
          assert.ok(analysis.nodes > 0);
          assert.ok(Number.isFinite(analysis.score));
        }
        game.undo(2);
        assert.equal(game.ply(), 22);
        assert.equal(JSON.parse(game.think(1, 10)).model, b === EngineModel.Light ? "light" : "lightning");
      } finally {
        game.free();
      }
    }
  }
});

test("replays preserve the models, seating, times and every move", () => {
  for (const a of ["light", "lightning"] as const) {
    for (const b of ["light", "lightning"] as const) {
      for (const mode of ["human", "ai"] as const) {
        for (const firstSeat of [0, 1] as const) {
          const record: GameRecord = {
            mode, firstSeat, models: [a, b], timeA: 500, timeB: 10000, seed: 0xffffffff,
            moves: [0, 31 | (31 << 5), 1 | (9 << 5), 30 | (29 << 5), 15 | (15 << 5)],
          };
          assert.deepEqual(decodeRecord(encodeRecord(record)), record);
        }
      }
    }
  }
});

test("replays reject unsupported versions", () => {
  assert.equal(decodeRecord(btoa(String.fromCharCode(255, 0, 0, 0, 0, 0))), null);
});
