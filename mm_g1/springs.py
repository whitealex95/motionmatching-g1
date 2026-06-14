"""Critically-damped spring + inertialization helpers.

Copied verbatim (math identical) from ../GenoViewPython-MotionMatching/genoview_g1.py,
which in turn follows Daniel Holden's spring/inertialization code. Used for the desired
trajectory prediction (TrajectorySpring*) and pose-transition blending (DecaySpring*).
"""
import numpy as np

from . import quat


def HalflifeToDamping(halflife, eps=1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def DecaySpringDamperPosition(x, v, halflife, dt):
    y = HalflifeToDamping(halflife) / 2.0
    j1 = v + x * y
    eydt = np.exp(-y * dt)
    return (eydt * (x + j1 * dt), eydt * (v - j1 * y * dt))


def TrajectorySpringPosition(pos, vel, acc, desiredVel, halflife, dt):
    y = HalflifeToDamping(halflife) / 2.0
    j0 = vel - desiredVel
    j1 = acc + j0 * y
    eydt = np.exp(-y * dt)
    return (
        eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) +
        (j1 / (y * y)) + j0 / y + desiredVel * dt + pos,
        eydt * (j0 + j1 * dt) + desiredVel,
        eydt * (acc - j1 * y * dt))


def TrajectorySpringRotation(rot, ang, desiredRot, halflife, dt):
    y = HalflifeToDamping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot, desiredRot)))
    j1 = ang + j0 * y
    eydt = np.exp(-y * dt)
    return (
        quat.mul(quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)), desiredRot),
        eydt * (ang - j1 * y * dt))
