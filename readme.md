# strategy-based-crowd-flow-dynamics
![Static Badge](https://img.shields.io/badge/status-done-brightgreen?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/type-research-purple?style=for-the-badge)

This repository contains the code behind our paper on strategy‑based crowd flow dynamics. The paper received the Minister’s Award at the 70th National Science Exhibition (Research; largest student contest in Korea) — 제70회 전국과학전람회 산업 및 에너지 부문 특상 (4위).

## Abstract

We present an agent-based crowd simulator that blends decision-level strategy with physical interaction modeling. Each pedestrian observes a limited field of view, reasons about nearby people and obstacles, and rotates a target heading using a perception-dependent “will” term. This intent is coupled with spring–damper style contact forces for human–human and human–obstacle interactions, enabling smooth avoidance as well as realistic collision responses. The model is calibrated against real-world trajectories: recorded tracks are projected into the ground plane, simulated paths are generated with matched spawn timing, and hyperparameters are optimized via dynamic time warping to minimize trajectory distance. The codebase reproduces the evaluation videos (crossroads, narrow passages, maze, obstacle courses, and real-world scenes) and provides scripts for running new simulations or parameter fitting on additional datasets.