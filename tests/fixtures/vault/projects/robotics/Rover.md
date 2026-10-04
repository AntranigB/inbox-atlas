---
title: Rover
tags: [robotics, rover, electronics]
date: 2026-02-11
---

# Rover chassis

The rover is a four wheel skid steer chassis cut from 3 mm aluminum plate. Each wheel has its own 12 V gear motor
with a hall encoder, driven by two dual channel motor driver boards. The battery is a 3S lithium pack with a
buck converter down to 5 V for the microcontroller and the servo rail on the [[Gripper]].

## Wiring notes

The motor driver enable pins go to PWM capable pins on the microcontroller. Encoder lines need pull up resistors or
the tick counts drift. Keep the motor ground and the logic ground joined at one star point near the battery.

## Next steps

Mount the arm on the front plate, add a bumper switch, and tune the wheel odometry so the rover drives a straight
two meter line without drifting more than five centimeters. #robotics
