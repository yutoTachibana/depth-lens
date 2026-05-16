# Playbook — depth-lens in production

Concrete recipes for the 3 production scenarios depth-lens is built for.

| Scenario | Recipe |
|---|---|
| **"Can we switch from Opus to Haiku?"** | [model-downgrade.md](./model-downgrade.md) |
| **"The vendor just shipped a new model — does it break us?"** | [regression-detection.md](./regression-detection.md) |
| **"We spend $5k/mo on Claude — where can we cut?"** | [cost-audit.md](./cost-audit.md) |

Each is one CLI invocation + how to interpret the output. None take more
than a few dollars or 10 minutes to run.
