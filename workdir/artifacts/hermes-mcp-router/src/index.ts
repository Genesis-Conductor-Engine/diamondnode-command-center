export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    try {
      if (path === "/health") {
        return jsonResponse({ status: "ok", service: "hermes-mcp-router", version: "1.0.0" });
      }

      if (path === "/mcp/hermes" && request.method === "POST") {
        return await handleMcpHermes(request, env);
      }

      if (path === "/mcp/hermes/ledger" && request.method === "GET") {
        return await handleGetLedger(env);
      }

      if (path === "/webhooks/pbc" && request.method === "POST") {
        return await handlePbcWebhook(request, env);
      }

      return jsonResponse({ error: "Not found" }, 404);
    } catch (err: any) {
      console.error("Unhandled error:", err);
      return jsonResponse({ error: "Internal server error", message: err.message }, 500);
    }
  },
};

async function handleMcpHermes(request: Request, env: Env): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  const { artifact, thermo_context, node_id, source } = body as any;

  if (!artifact || !node_id) {
    return jsonResponse({ error: "Missing required fields: artifact, node_id" }, 400);
  }

  const timestamp = new Date().toISOString();
  const entry: LedgerEntry = {
    timestamp,
    artifact_id: artifact.artifact_id || `unknown-${Date.now()}`,
    node_id,
    source: source || "unknown",
    tier: artifact.tiering?.model_tier || "unknown",
    cdi_score: artifact.tiering?.cdi_score ?? null,
    thermo_snapshot: thermo_context || {},
  };

  // Persist to KV if configured
  if (env.HERMES_LEDGER_KV) {
    const key = `ledger:${entry.timestamp}:${entry.node_id}`;
    await env.HERMES_LEDGER_KV.put(key, JSON.stringify(entry));
  }

  // Forward to PBC webhook if configured
  let pbcOk = false;
  if (env.PBC_WEBHOOK_URL) {
    try {
      const pbcPayload = {
        content: `Hermes MCP Router received artifact from ${node_id}`,
        embeds: [
          {
            title: "Hermes Artifact Ingest",
            description: `Artifact ${entry.artifact_id} from ${node_id} (tier: ${entry.tier})`,
            color: 0x00aaff,
            fields: [
              { name: "CDI Score", value: String(entry.cdi_score ?? "N/A"), inline: true },
              { name: "Source", value: entry.source, inline: true },
            ],
            timestamp: entry.timestamp,
          },
        ],
      };
      const pbcRes = await fetch(env.PBC_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pbcPayload),
      });
      pbcOk = pbcRes.ok;
    } catch (e) {
      console.error("PBC webhook failed:", e);
    }
  }

  // Compute wQFLOP-like metric (placeholder)
  const wqflop = computeWQFLOP(artifact);

  return jsonResponse({
    status: "accepted",
    artifact_id: entry.artifact_id,
    node_id,
    pbc_forwarded: pbcOk,
    wqflop,
    timestamp,
  });
}

async function handleGetLedger(env: Env): Promise<Response> {
  if (!env.HERMES_LEDGER_KV) {
    return jsonResponse({ error: "KV not configured" }, 503);
  }

  const list = await env.HERMES_LEDGER_KV.list({ prefix: "ledger:" });
  const entries: LedgerEntry[] = [];
  for (const key of list.keys) {
    const value = await env.HERMES_LEDGER_KV.get(key.name);
    if (value) entries.push(JSON.parse(value));
  }

  // Sort by timestamp desc
  entries.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

  return jsonResponse({ entries, count: entries.length });
}

async function handlePbcWebhook(request: Request, env: Env): Promise<Response> {
  if (!env.PBC_WEBHOOK_URL) {
    return jsonResponse({ error: "PBC_WEBHOOK_URL not configured" }, 503);
  }

  const payload = await request.json().catch(() => null);
  if (!payload) {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  const res = await fetch(env.PBC_WEBHOOK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return jsonResponse({
    forwarded: true,
    pbc_status: res.status,
    pbc_ok: res.ok,
  });
}

function computeWQFLOP(artifact: any): number {
  // Placeholder wQFLOP estimator based on CDI score and tier
  const tierMultipliers: Record<string, number> = {
    haiku: 1.0,
    sonnet: 2.5,
    opus: 5.0,
    mistral: 1.8,
    gpt4: 4.0,
  };
  const tier = artifact?.tiering?.model_tier || "unknown";
  const cdi = artifact?.tiering?.cdi_score || 0.5;
  const mult = tierMultipliers[tier] || 1.0;
  return parseFloat((cdi * mult * 1e12).toExponential(3));
}

function jsonResponse(data: any, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

interface LedgerEntry {
  timestamp: string;
  artifact_id: string;
  node_id: string;
  source: string;
  tier: string;
  cdi_score: number | null;
  thermo_snapshot: any;
}

interface Env {
  HERMES_LEDGER_KV?: KVNamespace;
  PBC_WEBHOOK_URL?: string;
  SLACK_WEBHOOK_URL?: string;
}
