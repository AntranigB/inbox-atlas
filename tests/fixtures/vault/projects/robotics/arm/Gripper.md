---
title: Gripper
tags: [robotics, arm, 3d-printing]
date: 2026-02-25
---

# Gripper design

The gripper is a two finger parallel claw printed in PETG. A single micro servo drives a rack and pinion so both
fingers close together. The fingers have TPU pads so the claw can hold an egg without cracking it.

## Grip force

At 6 V the micro servo stalls at about 1.8 kg cm, which gives roughly 4 newtons at the finger pads. That is enough
for a soda can but not a full water bottle. A metal gear servo would double it. Servo limits come from
[[Servo calibration]].
