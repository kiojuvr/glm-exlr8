# Project guidance

- Target only GLM-5.3 EXLR8 `up1` on M3 Ultra 512 GB initially.
- This repository contains planning artifacts and the bounded M1 experiment only. Implement subsequent milestones only when requested; keep experiment code separate from future production modules.
- Sibling `mlx-lm`, `MLXLR8`, `ds41f-mlx`, and `glm53-flash-mlx` repositories are read-only references.
- Read `docs/authority-and-findings.md` before adapting code. Preserve component copyrights, licenses, NOTICE files, source revisions, and modification notices when code is imported.
- Never label re-quantized MXFP4/affine4 weights as exact EXLR8. Keep approximate profiles explicit and off by default.
- Runtime owns one committed session frontier. Publish only after all state writes complete; no silent replay/repack recovery.
- Bound scratch and lazy graph lifetimes. A fixed-shape Python array is not proof of stable physical storage.
- Validate changed interfaces, not the entire upstream model repeatedly. Avoid recreating ds41f release/evidence machinery.
- Keep documentation facts, architectural inferences, and unmeasured hypotheses distinct.
