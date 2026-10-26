# Local (Ollama) vs hosted model for a plant copilot

Rough guidance, not measured in this repo — the numbers below are estimates to check per site.

| | Local (e.g. 7–8B model via Ollama on a site GPU/CPU box) | Hosted API |
|---|---|---|
| Data leaving the plant | none | prompts and SOP text go to a vendor |
| Works if internet drops | yes | no |
| Answer quality on long/ambiguous questions | lower | higher |
| Latency | depends on hardware (estimate: a few seconds on a modest GPU) | usually low, network dependent |
| Cost | hardware + upkeep, flat | per token, grows with use |

**Choice in this design:** the decision of *what* is allowed (refuse / escalate / tool / cite) is made by deterministic rules and validated schemas, not by the model. That keeps the model's job small — phrasing a cited SOP passage — which a local model handles well, so a local model is the default and a hosted model is optional for harder phrasing.
