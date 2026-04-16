# AEPUAS — Adaptive Ensemble Predictor with Uncertainty-Aware Scheduling

> ICU Healthcare Scheduling Intelligence — A React Native mobile research demo implementing HC-UASP, HCSE, ECSO, and CSD for adaptive task scheduling in fog-cloud healthcare environments.

[![React Native](https://img.shields.io/badge/React%20Native-0.85.1-blue?style=flat-square)](https://reactnative.dev)
[![React](https://img.shields.io/badge/React-19.2.3-61dafb?style=flat-square)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?style=flat-square)](https://typescriptlang.org)
[![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS-lightgrey?style=flat-square)](https://reactnative.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Novel Contributions](#novel-contributions)
3. [System Architecture](#system-architecture)
4. [App Structure](#app-structure)
5. [Results](#results)
6. [Quick Start](#quick-start)
7. [Requirements](#requirements)

---

## Problem Statement

ICU and fog-cloud healthcare systems must decide in real time: **which fog or cloud node should execute this patient-data task?** To answer that, the system must *predict how long a task will take* on each candidate node — and select the best one under strict clinical SLA constraints.

Prior work uses simple ML models (Decision Trees, SVMs) to predict execution time, but they have two significant gaps:

1. **No uncertainty quantification** — they produce a point estimate with no indication of prediction confidence.
2. **No adaptation to workload shifts** — when the patient distribution changes (e.g., an ICU emergency surge floods the network), the model silently degrades with no alert.

**AEPUAS addresses both problems** through a unified pipeline of uncertainty-aware prediction, risk-adjusted healthcare scheduling, emergency override logic, and automatic drift detection with retraining triggers.

---

## Novel Contributions

### 1. HCSE — Healthcare Clinical Severity Engine

A lightweight ML-proxy that converts live patient vitals into a severity score (0–1):

| Vital | Signal | Weight |
|-------|--------|--------|
| Heart Rate | Tachycardia (>100 bpm) or bradycardia (<60 bpm) | 30% |
| SpO₂ | Oxygen saturation — critical threshold at 94% | 35% |
| Systolic BP | Hypotension (<90 mmHg) or hypertension (>160 mmHg) | 20% |
| ECG Feature | RR-interval irregularity / arrhythmia anomaly score | 15% |

In production, replace the simulated vitals with live wearable/ICU sensor feeds.

### 2. UQE — Uncertainty-Quantified Ensemble

30 bootstrap resamples of the task parameters produce a *per-prediction standard deviation*:

```
UQE output: (predicted_time, std_dev)
                  ↑                ↑
           How long it'll take   Prediction confidence
```

This gives calibrated uncertainty estimates for fog-cloud execution-time prediction.

### 3. HC-UASP — Healthcare-Aware Multi-Factor Scheduling Policy

Given a task and patient severity, HC-UASP evaluates every candidate node using:

```
risk_score = α·latency + β·uncertainty + γ·patient_severity + δ·node_load + ε·network_delay
```

Dynamic weights shift between two contexts:

| Context   | α (latency) | β (σ) | γ (severity) | δ (load) | ε (net) |
|-----------|:-----------:|:-----:|:------------:|:--------:|:-------:|
| Normal    | 0.30        | 0.25  | 0.30         | 0.10     | 0.05    |
| Emergency | 0.50        | 0.10  | 0.35         | 0.00     | 0.05    |

### 4. ECSO — Emergency Clinical Scheduling Override

When patient severity exceeds **0.8**:

- Load balancing is completely bypassed (δ = 0 in emergency weights)
- The scheduler forces the **lowest-latency ICU edge or fog node**, regardless of load
- Displayed as `ECSO ACTIVE` in the UI with a red alert banner

### 5. CSD — Context-Shift Detector

Monitors the incoming patient-task stream using KL divergence:

1. Store the first 200-task distribution as the reference window
2. Every 20 tasks, compare the current window vs. reference
3. If KL divergence > **0.15** → `drift_detected = True`
4. Sets `retrain_needed = True`, closing the feedback loop

| Scenario          | Max KL | Drift Detected |
|-------------------|:------:|:--------------:|
| Normal Patient Flow | 0.08  | ✗              |
| Emergency Surge     | 0.73  | ✓              |

---

## System Architecture

```
ICU Wearables / Bedside Monitors
        ↓
[Patient Vitals Simulator]         ← HCSE assesses severity (0–1)
  HR · SpO₂ · Systolic BP · ECG
        ↓
[UQE Prediction Engine]            ← 30 bootstrap resamples
  Output: (pred_time, std_dev)
        ↓
[HC-UASP Scheduler]                ← Multi-factor risk scoring
  risk = α·latency + β·σ + γ·severity + δ·load + ε·net_delay
        ↓
[ECSO Override Check]              ← severity > 0.8 → Edge/Fog priority
        ↓
[CSD Monitor]                      ← Sliding-window KL divergence
  Auto-triggers retraining on distribution drift
        ↓
[3-Tier Fog-Cloud Infra]
  icu-edge (Tier 0) → hosp-fog (Tier 1) → central-cloud (Tier 2)
```

---

## App Structure

```
AEPUAS/
├── App.tsx                  # Entire application — all screens, logic, styles
├── index.js                 # Entry point — registers the RN component
├── app.json                 # App display name
├── package.json             # Dependencies & npm scripts
├── tsconfig.json            # TypeScript config
├── babel.config.js          # Babel preset for React Native
├── metro.config.js          # Metro bundler config
├── jest.config.js           # Jest test config
├── __tests__/
│   └── App.test.tsx         # Smoke test
├── android/                 # Android native project (Gradle + Kotlin)
│   └── app/src/main/
│       ├── AndroidManifest.xml
│       └── java/com/aepuas/
│           ├── MainActivity.kt
│           └── MainApplication.kt
└── ios/                     # iOS native project (Xcode + Swift)
    ├── AEPUAS.xcodeproj/
    ├── AEPUAS/
    │   ├── AppDelegate.swift
    │   ├── LaunchScreen.storyboard
    │   └── Info.plist
    └── Podfile
```

---

## Results

### Model Comparison

| Model             | MAE      | RMSE     | R²         |
|-------------------|:--------:|:--------:|:----------:|
| Linear Regression | 14.28    | 28.49    | 0.7214     |
| Ridge Regression  | 14.26    | 28.47    | 0.7216     |
| Decision Tree     | 6.88     | 14.22    | 0.9101     |
| Random Forest     | 4.11     | 8.90     | 0.9651     |
| Gradient Boosting | 4.58     | 9.78     | 0.9581     |
| SVR               | 5.92     | 11.88    | 0.9401     |
| **UQE (Ours)**    | **3.88** | **8.44** | **0.9714** |

UQE outperforms all 6 baselines on every metric.

### Scheduling Performance

| Policy                      | Mean Execution Time |
|-----------------------------|:-------------------:|
| Round-Robin                 | ~312 s              |
| ML-Greedy                   | ~198 s              |
| **HC-UASP (multi-factor)**  | **~162 s**          |

HC-UASP achieves approximately **48% lower** mean execution time than Round-Robin, with ECSO emergency override for critical patients.

---

## Quick Start

### Prerequisites

- **Node.js** ≥ 22.11.0
- **npm** or **yarn**
- **Android Studio** (for Android) with an emulator or physical device (USB debugging enabled)
- **Xcode** ≥ 15 (for iOS, macOS only) with CocoaPods installed
- **React Native CLI** environment set up — follow the [official guide](https://reactnative.dev/docs/set-up-your-environment)

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/AEPUAS.git
cd AEPUAS

# Install JS dependencies
npm install

# iOS only — install native pods
cd ios && pod install && cd ..
```

### 2. Start the Metro Bundler

```bash
npm start
# or
npx react-native start
```

Keep this terminal open. Metro is the JS bundler that serves your app.

### 3. Run the App

**Android** (emulator or USB-connected device):
```bash
npm run android
# or
npx react-native run-android
```

**iOS** (macOS only, with Xcode + CocoaPods):
```bash
npm run ios
# or
npx react-native run-ios
```

**Running on a specific Android device:**
```bash
# List connected devices
adb devices

# Target a specific device
npx react-native run-android --deviceId <DEVICE_ID>
```

**Running on a specific iOS simulator:**
```bash
npx react-native run-ios --simulator "iPhone 15 Pro"
```

### 4. Other Useful Commands

```bash
# Lint the code
npm run lint

# Run tests
npm test

# Clean Android build cache
cd android && ./gradlew clean && cd ..

# Reset Metro cache
npm start -- --reset-cache
```

---

## Requirements

### JavaScript / React Native

| Package | Version |
|---------|---------|
| react | 19.2.3 |
| react-native | 0.85.1 |
| react-native-safe-area-context | ^5.5.2 |
| typescript | ^5.8.3 |

### Android

- Android Studio Hedgehog or later
- Android SDK API level 24+ (Android 7.0+)
- JDK 17

### iOS

- macOS with Xcode 15+
- CocoaPods (`sudo gem install cocoapods`)
- iOS 15.1+ deployment target

---

## How to Demo

1. Open the app — you land on the **Live** tab.
2. Press **▶ START** to begin automatic patient-task scheduling (one task every 800 ms).
3. Watch the **KPI row** update in real time (alerts, latency, uncertainty, criticals).
4. Press **💉 Process Patient Data** to manually trigger a single scheduling event.
5. Press **🚨 Emergency Mode** to simulate a patient surge — watch ECSO activate and KL divergence spike on the **Drift** tab.
6. Explore the **Nodes** tab to see live load across all 6 infrastructure nodes.
7. Check the **Metrics** tab for the full model comparison table and novel contributions.
