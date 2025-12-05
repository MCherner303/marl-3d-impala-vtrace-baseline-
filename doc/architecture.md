# Architecture Overview (Phase-1)

**Goal:** robust, production-style MARL training substrate.

- **Actors → Learner (IMPALA):** many actors stream unrolls to a centralized learner.  
- **Off-policy correction:** **V-trace** targets blended with **TD(λ)**; **EMA target** network for the critic; cross-update EMA for additional smoothing.  
- **Critic stability:** removed per-batch circular normalization; value-only smoothing; separate optimizer schedule for value head.  
- **Behavioral shaping:** distance-scaled feature + learnable **monotone prior** on drop logits → PRE monotonicity guaranteed by design.  
- **Health checks:** KL/entropy, off-policy clipping, legality audits, monotonicity, EV.

![dataflow](../results/plots/ev_curve_with_markers.png)

*(Figure reused here as a placeholder; in Phase-2 we'll add a real diagram.)*
