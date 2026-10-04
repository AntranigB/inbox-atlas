---
title: Servo calibration
tags: [robotics, arm, servo]
date: 2026-02-18
---

# Servo calibration

Every joint of the arm uses a hobby servo, and no two servos map pulse width to angle the same way. Calibration
means finding the pulse width for 0 and 180 degrees on each joint and storing it in a small table on the
microcontroller.

## Pulse widths

Shoulder servo: 540 microseconds at 0 degrees, 2410 microseconds at 180 degrees. Elbow servo: 610 to 2380
microseconds. Wrist servo: 500 to 2450 microseconds. The elbow buzzes past 170 degrees, so the software limit is
165 degrees for that joint.

## Procedure

Sweep each servo slowly with a potentiometer, mark the hard stops with a protractor, then back off two degrees from
each stop. Recalibrate after any crash, because the servo horn can slip a spline. See [[Gripper]] for the claw servo.
