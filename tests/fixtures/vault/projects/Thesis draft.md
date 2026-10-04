---
title: Thesis Draft
tags: [thesis, robotics, tactile-sensors, project-log]
date: 2025-03-12
---

# Thesis Draft

I spent most of the morning reorganizing my literature folder and trying to decide on a consistent naming scheme for all the PDFs. It's amazing how quickly things get messy when you're downloading one paper after another. After that I brewed another cup of coffee and sat down to actually make some progress on the thesis note itself. The weather outside is still gray and drizzly, which somehow makes it easier to stay focused indoors.

## Goal

The core goal of this senior thesis is to develop and characterize a set of low-cost tactile sensors suitable for mounting on robot grippers. Instead of relying on expensive commercial solutions that can cost hundreds of dollars per sensor, I'm exploring piezoresistive fabric, barometric pressure sensors, and simple capacitive designs that can be fabricated in the university makerspace for under $15 each. The ultimate aim is to show that these budget sensors can still deliver usable force and contact location data for basic grasping tasks. If successful, the work could open the door for more accessible robotics education and small-scale automation projects.

## Milestones

- Complete draft of Chapter 2 (Related Work) and deliver to advisor Prof. Okafor by **March 20**.
- Finish hardware assembly and initial calibration of the gripper prototype by end of March.
- Collect preliminary grasping dataset using the tactile array on at least three different objects.
- Write Chapters 3 and 4 (Methods and Results) during the first two weeks of April.
- Final thesis submission deadline: **April 24**.

I'm currently a little behind on the literature review because I keep finding new papers that feel relevant. The March 20 deadline for Chapter 2 is starting to feel closer than I'd like.

## Status

The sensor boards themselves are working better than expected after the last firmware tweak. I can now stream pressure and contact data at 80 Hz without dropping packets, which is plenty for most grasping experiments. The 3D-printed gripper fingers have been mounted on the WidowX arm in the lab and everything moves smoothly. However, the force calibration rig that I built last semester is currently out of commission. The load cell started giving erratic readings two weeks ago and eventually failed completely. Without a reliable way to map raw sensor values to actual Newtons, all my force data is basically useless. I ordered a replacement load cell from DigiKey on **March 4** and it should arrive any day now. Once it gets here I'll rebuild the calibration fixture and re-run the characterization sweeps.

I'm also about halfway through drafting Chapter 2. So far I've covered commercial tactile solutions, recent academic work on fabric-based sensors, and a few papers on optical tactile arrays. Still need to add a section comparing power consumption and repeatability across the different approaches.

## Risks

Biggest risk right now is the broken force calibration rig. If the new load cell from DigiKey is delayed or doesn't work with my existing amplifier circuit, I could lose an entire week of testing time. That would put serious pressure on the April 24 final submission deadline. Secondary risks include the robot arm's ROS driver crashing during long data-collection runs (happened twice last week) and the possibility that the piezoresistive fabric degrades faster than expected under repeated loading. I'm keeping a close eye on both.

## Log

- 2025-03-12: Updated this master thesis note, reorganized literature folder, wrote first draft of the Related Work comparison table.
- 2025-03-10: Discovered calibration rig failure during evening test session. Spent two hours debugging before concluding the load cell is dead.
- 2025-03-04: Placed DigiKey order for replacement 5 kg load cell (part number 3134_0 or equivalent). Tracking says it shipped yesterday.
- 2025-03-01: Finished firmware updates to support simultaneous sampling from all eight taxels.
- 2025-02-22: First successful grasp trial with the new sensor array. Could clearly detect slip on a plastic cup.

I need to remember to back up the latest dataset to the lab server before I leave today. Also should schedule a meeting with Prof. Okafor sometime next week to go over the current draft of Chapter 2 before the March 20 deadline. The semester is flying by faster than I expected.

(Word count: 682)
