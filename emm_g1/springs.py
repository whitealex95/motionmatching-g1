"""Critically-damped spring helpers for trajectory prediction and inertialization.

Ported from the GenoViewPython motion-matching demo (Daniel Holden's spring
formulation). ``Trajectory*`` predict where a desired position/rotation will be a
few horizons into the future; ``DecaySpringDamper*`` decay an inertialization
offset smoothly to zero so pose transitions don't pop.

Unity (com.jlpm.motionmatching) replaces ``exp(-x)`` everywhere with the rational
approximation ``FastNEgeExp(x) = 1/(1 + x + 0.48 x^2 + 0.235 x^3)``; we match it
exactly here so the predicted trajectories / decay are bit-for-bit equivalent.
"""

import numpy as np

from . import quat

_LN2_x4 = 4.0 * 0.69314718056


def halflife_to_damping(halflife, eps=1e-5):
    return _LN2_x4 / (halflife + eps)


def fast_neg_exp(x):
    """Unity ``FastNEgeExp`` rational approximation of ``exp(-x)``."""
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def decay_spring_damper(x, v, halflife, dt):
    """Decay an offset (x, v) toward zero with the given half-life."""
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y
    eydt = fast_neg_exp(y * dt)
    return eydt * (x + j1 * dt), eydt * (v - j1 * y * dt)


def trajectory_spring_position(pos, vel, acc, desired_vel, halflife, dt):
    """Predict (pos, vel, acc) at time ``dt`` given a desired velocity target."""
    y = halflife_to_damping(halflife) / 2.0
    j0 = vel - desired_vel
    j1 = acc + j0 * y
    eydt = fast_neg_exp(y * dt)
    return (
        eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y))
        + (j1 / (y * y)) + j0 / y + desired_vel * dt + pos,
        eydt * (j0 + j1 * dt) + desired_vel,
        eydt * (acc - j1 * y * dt),
    )


def trajectory_spring_rotation(rot, ang, desired_rot, halflife, dt):
    """Predict (rot, ang) at time ``dt`` given a desired rotation target."""
    y = halflife_to_damping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot, desired_rot)))
    j1 = ang + j0 * y
    eydt = fast_neg_exp(y * dt)
    return (
        quat.mul(quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)), desired_rot),
        eydt * (ang - j1 * y * dt),
    )
