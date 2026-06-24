"""Environment-aware Motion Matching (EMM) for the Unitree G1.

An *additive*, backward-compatible variant of the ``mm_g1`` motion matcher that
implements Ponton et al. 2025, "Environment-aware Motion Matching" (ACM TOG
44(6), Article 232). Instead of a manual ``J`` trigger, a low obstacle in the
scene makes the G1 hop over it automatically: each candidate pose carries a
ground-plane footprint ellipse and a vertical height range at its future
trajectory taps, and the search adds a log-barrier obstacle *penalization* that
gates out poses whose body would collide with the obstacle's height band. A jump
clip's airborne phase clears a low bar, so the env-aware search naturally selects
it -- no trigger, no tagging.

Nothing in ``mm_g1`` is modified; this package reuses the *same dataset*
(``data/gmr_lafan1_g1`` locomotion + ``data/g1_jump`` jumps) via its own database
build. Jump is the only environment target (height-gated, ``height_mode=True``).
"""
