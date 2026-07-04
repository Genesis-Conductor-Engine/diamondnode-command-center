import { Hono } from "hono";
import { start } from "workflow/api";
import { runThermoEpochPipeline } from "../workflows/thermo-epoch.js";

const app = new Hono();

app.get("/health", (c) =>
  c.json({ status: "ok", service: "thermo-workflow", workflow: "opux-epoch" }),
);

app.post("/api/thermo-epoch", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const variant = (body as { variant?: string }).variant ?? "chr17:41234470:A>G";
  const chain = (body as { chain?: string }).chain ?? "base";
  await start(runThermoEpochPipeline, [variant, chain]);
  return c.json({ message: "Thermo epoch workflow started", variant, chain });
});

export default app;