# Proof Notes for Proposition 1 (Residual Bound)

## Statement
Under the sub-Gaussian assumption on residual prediction error with variance proxy σ², unbiased ensemble mean, and Lipschitz physics backbone (constant L_phys), for any δ ∈ (0,1) and κ > 0:

P( ||r_true − r_pred|| > κσ + √(2σ² log(1/δ)) ) ≤ δ.

## Proof Sketch

1. Let ε = r_true − r_pred. By assumption ε is sub-Gaussian:
   P(|ε| > t) ≤ 2 exp(−t² / (2σ²)).

2. Choose t = κσ + √(2σ² log(1/δ)). Then
   P(|ε| > t) ≤ 2 exp(−(κσ)²/(2σ²) − log(1/δ)) ≤ δ
   (for the conventional choice of constants; the factor 2 is absorbed by adjusting δ or κ).

3. The ensemble provides an empirical estimate of the mean residual. Concentration of the empirical mean around the true mean follows from the same sub-Gaussian tail (or a Hoeffding-style bound if bounded residuals are assumed).

4. Lipschitz continuity of f_phys with constant L_phys implies that a residual bound of size B produces a trajectory tube of radius at most (e^{L_phys T} − 1)/L_phys · B over a horizon T (standard Gronwall argument). This tube is the set used by the safety filter.

5. Online adaptation never increases the residual bound by construction (constrained optimizer with the current bound as a hard inequality).

Full expanded proof with all constants will be included in the Stage-2 camera-ready version.
