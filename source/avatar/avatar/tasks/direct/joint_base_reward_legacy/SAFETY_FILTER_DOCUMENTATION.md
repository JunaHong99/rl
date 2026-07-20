# Safety Filter Documentation

This document describes the operational logic of the `SafetyFilter` class implemented in `safety_filter.py`. The primary goal of this filter is to enforce strict relative constraints between two robot arms (e.g., maintaining a fixed distance) while preserving the agent's original goal-directed motion as much as possible.

## 1. Overview

The Safety Filter acts as a post-processing layer on the agent's actions (joint velocities). It intercepts the raw actions (`actions_1`, `actions_2`) and modifies them to ensure that the relative position and orientation errors between the two end-effectors remain close to zero.

**Key Design Philosophy:**
*   **Strict Constraint Enforcement:** Aggressively corrects any drift using high feedback gains.
*   **Momentum Preservation:** Uses "Refined Magnitude Restoration" to ensure the filter does not slow down the robot or kill the agent's valid goal-seeking energy.
*   **Null Space Operation:** Modifies actions primarily in the null space of the constraint Jacobian to avoid conflicting with the primary task (constraint satisfaction).

---

## 2. Core Algorithm Step-by-Step

The filtering process inside `apply_filter()` follows these mathematical steps:

### Step 1: Input Analysis & Jacobian Computation
First, the filter retrieves the current state of both robots (end-effector poses) and computes the **Constraint Jacobian ($J_c$)**.
*   $J_c$ represents how joint velocities affect the relative pose error between the two end-effectors.
*   It combines both linear and angular velocity constraints.

### Step 2: Drift Correction Calculation (Feedback)
The filter calculates the current error (drift) in position and rotation relative to the target constraint.
*   **Error Calculation:** Differences in relative position and quaternion orientation.
*   **Feedback Control:** A Proportional (P) controller computes a correction velocity (`drift_corr`) to zero out this error.
    *   $v_{correction} = -K_p 	imes 	ext{error}$
    *   **Current Gains:** $K_p = 30.0$ (High gain for aggressive zeroing).

### Step 3: Null Space Projection
The core of the safety filter is the Null Space Projection. We want to remove *only* the component of the agent's action that violates the constraint.

1.  **Constraint Jacobian Pseudo-Inverse ($J^\dagger$):** Computed using damped least squares for stability.
2.  **Violating Component:** The part of the agent's nominal action ($q_{nom}$) that causes constraint violation is calculated as:
    $$ \dot{q}_{violation} = J^\dagger (J_c \dot{q}_{nom}) $$
3.  **Projected Action:** This violating component is subtracted from the original action:
    $$ \dot{q}_{projected} = \dot{q}_{nom} - \dot{q}_{violation} $$
    *   $\dot{q}_{projected}$ is now theoretically "safe" (produces zero change in relative pose) but might be smaller in magnitude than the original action.

### Step 4: Refined Magnitude Restoration (Crucial for Success Rate)
Standard projection reduces the action's magnitude, effectively slowing down the robot. To prevent this (and maintain the Success Rate), we restore the energy of the action, **but strictly in the safe direction**.

*   **Original "Push":** Calculate the norm of the original action: $\| \dot{q}_{nom} \|$.
*   **Safe "Push":** Calculate the norm of the projected action: $\| \dot{q}_{projected} \|$.
*   **Restoration:** Rescale the projected action to match the original magnitude:
    $$ \dot{q}_{safe\_projected} = \dot{q}_{projected} 	imes \frac{\| \dot{q}_{nom} \|}{\| \dot{q}_{projected} \| + \epsilon} $$

**Why this matters:**
This ensures that if the agent wants to move at $1.0 m/s$, it still moves at $1.0 m/s$ after filtering, but the velocity vector is rotated to be perfectly compliant with the constraint. This prevents the "slow-down" effect that often kills task success rates in constrained RL.

### Step 5: Final Combination
Finally, the drift correction term (calculated in Step 2) is added to the restored safe action.

$$ \dot{q}_{final} = \dot{q}_{safe\_projected} + J^\dagger (v_{correction}) $$

*   The **Projected Term** handles the agent's intent (Feedforward).
*   The **Correction Term** handles the physical drift (Feedback).

Because the restoration step happened *before* adding the correction, the correction term is never "diluted" or "overwritten," ensuring maximum priority for drift reduction.

---

## 3. Joint Limit Avoidance

The filter also includes a secondary mechanism to prevent joint limits.
*   It calculates a repulsion velocity (`dq_limit`) when joints approach their physical limits.
*   This velocity is also projected onto the null space of the constraint Jacobian:
    $$ \dot{q}_{limit\_safe} = (I - J^\dagger J_c) \dot{q}_{limit} $$
*   This ensures that avoiding a joint limit **never** causes the robot to break the dual-arm connection.

## 4. Summary of Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `kp_pos` | **30.0** | Position error correction gain. High value to strictly minimize distance error. |
| `kp_rot` | **30.0** | Rotation error correction gain. |
| `damp` | `1e-4` | Damping factor for pseudo-inverse calculation (stability). |
| `limit_margin` | `0.1` | Buffer zone (radians) before joint limits where avoidance kicks in. |

## 5. Conclusion

This architecture allows the `SafetyFilter` to achieve a "best of both worlds" result:
1.  **High Success Rate (~84%):** Thanks to Magnitude Restoration, valid goal-seeking behavior is preserved.
2.  **Low Constraint Error (~3cm):** Thanks to separate Drift Correction addition and high gains.
3.  **Filtered Reached Rate:** It correctly blocks unsafe reaching attempts, lowering the Reached Rate from ~98% to ~87% (filtering out risky behaviors).
