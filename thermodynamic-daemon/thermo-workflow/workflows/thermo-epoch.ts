import { sleep } from "workflow";

export async function runThermoEpochPipeline(variant: string, chain: string = "base") {
  "use workflow";

  const thermo = await fetchThermoState();
  const epoch = await runOpuxEpoch(variant, thermo);
  await sleep("2s");
  const story = await writeAlchemyStory(epoch, chain);
  const witness = await buildAttestationWitness(epoch, story);
  await pushToKv(epoch, witness);

  return {
    epoch_id: epoch.epoch_id,
    work_hash: epoch.work_hash,
    reducibility_score: epoch.reducibility_score,
    story_id: story.story_id,
  };
}

async function fetchThermoState() {
  "use step";
  const res = await fetch("http://127.0.0.1:9100/state");
  if (!res.ok) return { temperature_c: 55, vram: { used_pct: 30 } };
  return res.json();
}

async function runOpuxEpoch(variant: string, _thermo: Record<string, unknown>) {
  "use step";
  const { spawnSync } = await import("node:child_process");
  const { readFileSync } = await import("node:fs");
  const proc = spawnSync(
    "/home/diamondnode/venv312/bin/python",
    ["/home/diamondnode/thermodynamic-daemon/epoch_orchestrator.py", "--variant", variant, "--no-alphagenome"],
    { encoding: "utf-8", timeout: 120000 },
  );
  if (proc.status !== 0) throw new Error(proc.stderr || "epoch failed");
  const epoch = JSON.parse(readFileSync("/tmp/thermo-epoch/epoch_latest.json", "utf-8"));
  return epoch as Record<string, unknown>;
}

async function writeAlchemyStory(epoch: Record<string, unknown>, chain: string) {
  "use step";
  const { spawnSync } = await import("node:child_process");
  const proc = spawnSync(
    "/home/diamondnode/venv312/bin/python",
    ["/home/diamondnode/thermodynamic-daemon/story_from_epoch.py", "-", chain],
    { input: JSON.stringify(epoch), cwd: "/home/diamondnode/thermodynamic-daemon", encoding: "utf-8" },
  );
  if (proc.status !== 0) throw new Error(proc.stderr || "story log failed");
  return JSON.parse(proc.stdout.trim());
}

async function buildAttestationWitness(epoch: Record<string, unknown>, story: Record<string, unknown>) {
  "use step";
  const { writeFileSync, unlinkSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const tmp = join(tmpdir(), `thermo-witness-${Date.now()}.json`);
  writeFileSync(tmp, JSON.stringify({ epoch, story }));
  const { spawnSync } = await import("node:child_process");
  const proc = spawnSync(
    "/home/diamondnode/venv312/bin/python",
    ["-c", "import json,sys; from epoch_orchestrator import build_attestation_witness; d=json.load(open(sys.argv[1])); print(json.dumps(build_attestation_witness(d['epoch'],d['story'])))", tmp],
    { cwd: "/home/diamondnode/thermodynamic-daemon", encoding: "utf-8" },
  );
  try { unlinkSync(tmp); } catch { /* ignore */ }
  if (proc.status !== 0) throw new Error(proc.stderr || "witness failed");
  return JSON.parse(proc.stdout.trim());
}

async function pushToKv(epoch: Record<string, unknown>, witness: Record<string, unknown>) {
  "use step";
  const { spawnSync } = await import("node:child_process");
  spawnSync("/home/diamondnode/bin/coalition-kv-push.sh", { encoding: "utf-8", timeout: 60000 });
  return { ok: true, epoch_id: epoch.epoch_id, witness_type: witness.type };
}