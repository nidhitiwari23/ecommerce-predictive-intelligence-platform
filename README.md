<div align="center">

<!-- HERO BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2d5e,50:1e56a0,100:14b8a6&height=200&section=header&text=E-Commerce%20Predictive%20Intelligence&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=Industry-Grade%20ML%20Platform%20%7C%20End-to-End%20Data%20Science&descAlignY=58&descSize=16&animation=fadeIn"/>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://xgboost.ai)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **An industry-grade, end-to-end machine learning platform that transforms raw e-commerce transactions into predictive intelligence — featuring 6 production ML models, real-time API serving, automated drift monitoring, and interactive dashboards.**

<br/>

[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp; [📊 Results](#-model-results) &nbsp;·&nbsp; [🏗️ Architecture](#%EF%B8%8F-system-architecture) &nbsp;·&nbsp; [📁 File Guide](#-complete-file-guide) &nbsp;·&nbsp; 

</div>

---

## ✨ What This Platform Does

Most businesses can see **what happened yesterday**. This platform tells them **what will happen tomorrow**.

<br/>

<div align="center">

| 🔴 The Problem | 🟢 Our Solution | 📈 Business Impact |
|:---|:---|:---|
| Can't predict which customers will leave | XGBoost Churn Model (AUC **0.891**) | Retain customers before they leave  |
| Manual demand forecasting is 20–25% wrong | Prophet + LSTM Hybrid (MAPE **5.9%**) | Right inventory levels → fewer stockouts & wastage |
| No idea who your best customers are | K-Means Segmentation  | Targeted marketing to the right segment |
| Fraud detection is manual and slow | Isolation Forest Anomaly Detection | Flags suspicious transactions automatically |
| Models degrade silently after deployment | Evidently AI Drift Monitoring | Weekly alerts before accuracy drops |

</div>

---

## 🏗️ System Architecture

> The complete data flow from raw CSV files to live predictions and dashboards.

<br/>

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-Commerce Predictive Intelligence — Architecture Diagrams</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --navy:    #060d1f;
    --panel:   #0d1b35;
    --card:    #0f2348;
    --border:  #1e3a6a;
    --blue:    #2563eb;
    --lblue:   #3b82f6;
    --bblue:   #60a5fa;
    --teal:    #0d9488;
    --lteal:   #14b8a6;
    --green:   #16a34a;
    --lgreen:  #22c55e;
    --amber:   #d97706;
    --lamber:  #f59e0b;
    --rose:    #e11d48;
    --lrose:   #f43f5e;
    --purple:  #7c3aed;
    --lpurple: #a78bfa;
    --white:   #f8fafc;
    --text:    #cbd5e1;
    --muted:   #64748b;
    --glow:    rgba(59,130,246,0.18);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--navy);
    color: var(--white);
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Background grid ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(37,99,235,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(37,99,235,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .page-wrap { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }

  /* ── Page header ── */
  .page-header {
    text-align: center;
    margin-bottom: 60px;
    padding: 48px 0 40px;
  }
  .page-header .badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: var(--bblue);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
  }
  .page-header h1 {
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #f8fafc 0%, var(--bblue) 60%, var(--lteal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .page-header p {
    color: var(--text);
    font-size: 16px;
    font-weight: 300;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.7;
  }

  /* ── Section titles ── */
  .section-label {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    margin-top: 60px;
  }
  .section-label .num {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: var(--blue);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700; color: white;
    flex-shrink: 0;
  }
  .section-label h2 { font-size: 22px; font-weight: 700; color: var(--white); }
  .section-label p  { font-size: 13px; color: var(--muted); margin-top: 2px; }

  /* ── Diagram canvas ── */
  .diagram-canvas {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 32px;
    position: relative;
    overflow: hidden;
  }
  .diagram-canvas::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%);
    pointer-events: none;
  }

  /* ═══════════════════════════════════════════════
     ARCHITECTURE DIAGRAM COMPONENTS
  ══════════════════════════════════════════════ */

  /* Layer blocks */
  .layer {
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 0;
    overflow: hidden;
  }
  .layer-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .layer-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
  .layer-body { padding: 16px 20px 20px; }

  /* Color variants */
  .layer-data   .layer-header { background: rgba(13,148,136,0.15); color: var(--lteal); border-bottom: 1px solid rgba(13,148,136,0.25); }
  .layer-data   .layer-header .dot { background: var(--lteal); box-shadow: 0 0 8px var(--lteal); }
  .layer-data   { border-color: rgba(13,148,136,0.3); }

  .layer-pipe   .layer-header { background: rgba(37,99,235,0.15); color: var(--bblue); border-bottom: 1px solid rgba(37,99,235,0.25); }
  .layer-pipe   .layer-header .dot { background: var(--bblue); box-shadow: 0 0 8px var(--bblue); }
  .layer-pipe   { border-color: rgba(37,99,235,0.3); }

  .layer-ml     .layer-header { background: rgba(124,58,237,0.15); color: var(--lpurple); border-bottom: 1px solid rgba(124,58,237,0.25); }
  .layer-ml     .layer-header .dot { background: var(--lpurple); box-shadow: 0 0 8px var(--lpurple); }
  .layer-ml     { border-color: rgba(124,58,237,0.3); }

  .layer-mlops  .layer-header { background: rgba(217,119,6,0.15); color: var(--lamber); border-bottom: 1px solid rgba(217,119,6,0.25); }
  .layer-mlops  .layer-header .dot { background: var(--lamber); box-shadow: 0 0 8px var(--lamber); }
  .layer-mlops  { border-color: rgba(217,119,6,0.3); }

  .layer-output .layer-header { background: rgba(22,163,74,0.15); color: var(--lgreen); border-bottom: 1px solid rgba(22,163,74,0.25); }
  .layer-output .layer-header .dot { background: var(--lgreen); box-shadow: 0 0 8px var(--lgreen); }
  .layer-output { border-color: rgba(22,163,74,0.3); }

  /* Cards inside layers */
  .cards-row {
    display: grid;
    gap: 12px;
  }
  .cards-row.cols-2 { grid-template-columns: repeat(2, 1fr); }
  .cards-row.cols-3 { grid-template-columns: repeat(3, 1fr); }
  .cards-row.cols-4 { grid-template-columns: repeat(4, 1fr); }
  .cards-row.cols-5 { grid-template-columns: repeat(5, 1fr); }

  .arch-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    cursor: default;
  }
  .arch-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }
  .arch-card .card-icon { font-size: 22px; margin-bottom: 8px; }
  .arch-card .card-title {
    font-size: 13px; font-weight: 700;
    margin-bottom: 4px;
    color: var(--white);
  }
  .arch-card .card-algo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 8px;
    line-height: 1.5;
  }
  .arch-card .card-metric {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
  }

  /* Metric badge colours */
  .m-green  { background: rgba(22,163,74,0.2);  color: var(--lgreen);  border: 1px solid rgba(22,163,74,0.3);  }
  .m-blue   { background: rgba(37,99,235,0.2);  color: var(--bblue);  border: 1px solid rgba(37,99,235,0.3);  }
  .m-purple { background: rgba(124,58,237,0.2); color: var(--lpurple);border: 1px solid rgba(124,58,237,0.3); }
  .m-amber  { background: rgba(217,119,6,0.2);  color: var(--lamber); border: 1px solid rgba(217,119,6,0.3);  }
  .m-teal   { background: rgba(13,148,136,0.2); color: var(--lteal);  border: 1px solid rgba(13,148,136,0.3); }
  .m-rose   { background: rgba(225,29,72,0.2);  color: var(--lrose);  border: 1px solid rgba(225,29,72,0.3);  }

  /* Pipe step (inside pipeline layer) */
  .pipe-steps { display: flex; align-items: stretch; gap: 0; }
  .pipe-step {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 10px;
    padding: 14px;
    position: relative;
  }
  .pipe-step:not(:last-child)::after {
    content: '→';
    position: absolute;
    right: -16px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--bblue);
    font-size: 18px;
    font-weight: 700;
    z-index: 2;
    background: var(--panel);
    padding: 2px 4px;
    border-radius: 4px;
  }
  .pipe-step + .pipe-step { margin-left: 20px; }
  .pipe-step .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--bblue);
    font-weight: 600;
    margin-bottom: 5px;
  }
  .pipe-step .step-file {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--bblue);
    font-weight: 600;
    margin-bottom: 5px;
    background: rgba(37,99,235,0.12);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
  }
  .pipe-step .step-title { font-size: 12px; font-weight: 700; color: var(--white); margin-bottom: 6px; }
  .pipe-step .step-desc  { font-size: 11px; color: var(--text); line-height: 1.5; }
  .pipe-step .step-io    { margin-top: 8px; }
  .pipe-step .step-io span {
    display: block;
    font-size: 10px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
  }
  .step-io span b { color: var(--bblue); }

  /* Connector arrow between layers */
  .connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    margin: 8px 0;
  }
  .connector .arrow-line {
    width: 2px;
    height: 28px;
    background: linear-gradient(to bottom, var(--border), var(--blue));
  }
  .connector .arrow-head {
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid var(--blue);
  }
  .connector .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    background: var(--panel);
    padding: 2px 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin: 4px 0;
  }

  /* Dataset card (top layer) */
  .dataset-card {
    background: rgba(13,148,136,0.08);
    border: 1px solid rgba(13,148,136,0.3);
    border-radius: 10px;
    padding: 16px 18px;
    flex: 1;
  }
  .dataset-card .ds-title {
    font-size: 13px; font-weight: 700; color: var(--lteal); margin-bottom: 6px;
  }
  .dataset-card .ds-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--muted); line-height: 1.7;
  }
  .dataset-card .ds-tag {
    display: inline-block;
    background: rgba(13,148,136,0.15);
    color: var(--lteal);
    border: 1px solid rgba(13,148,136,0.3);
    font-size: 10px;
    padding: 1px 7px;
    border-radius: 8px;
    margin-top: 6px;
    margin-right: 4px;
  }

  /* MLOps + Output side-by-side */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* ═══════════════════════════════════════════════
     DATA FLOW DIAGRAM
  ══════════════════════════════════════════════ */
  .flow-wrap { display: flex; flex-direction: column; gap: 0; align-items: center; }

  .flow-step {
    width: 100%;
    max-width: 820px;
    background: var(--card);
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    transition: box-shadow 0.25s;
    position: relative;
  }
  .flow-step:hover { box-shadow: 0 0 0 2px rgba(59,130,246,0.35), 0 12px 32px rgba(0,0,0,0.35); }

  .flow-step-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
  }
  .flow-num {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800;
    flex-shrink: 0;
  }
  .flow-step-header .flow-title { font-size: 14px; font-weight: 700; color: var(--white); }
  .flow-step-header .flow-file  {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 6px;
    background: rgba(37,99,235,0.12);
    color: var(--bblue);
    border: 1px solid rgba(37,99,235,0.25);
  }

  .flow-step-body {
    display: grid;
    grid-template-columns: 1fr 2px 1fr 2px 1fr;
    gap: 0;
  }
  .flow-col {
    padding: 14px 16px;
  }
  .flow-divider { background: var(--border); }
  .flow-col-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .flow-col ul { list-style: none; }
  .flow-col ul li {
    font-size: 12px;
    color: var(--text);
    padding: 3px 0;
    display: flex;
    align-items: flex-start;
    gap: 6px;
    line-height: 1.4;
  }
  .flow-col ul li::before {
    content: '›';
    color: var(--bblue);
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
  }
  .flow-col .out-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 5px;
    margin: 3px 3px 3px 0;
    background: rgba(22,163,74,0.12);
    color: var(--lgreen);
    border: 1px solid rgba(22,163,74,0.25);
  }

  /* Arrow between flow steps */
  .flow-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 6px 0;
    gap: 0;
  }
  .flow-arrow .fa-line {
    width: 2px;
    height: 18px;
    background: linear-gradient(to bottom, var(--border), var(--blue));
  }
  .flow-arrow .fa-arrow {
    width: 0; height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 9px solid var(--blue);
  }

  /* Split arrows (Step 4 branches) */
  .flow-branch {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    width: 100%;
    max-width: 820px;
    padding: 6px 0;
    position: relative;
  }
  .flow-branch::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue), var(--purple), var(--blue), transparent);
    opacity: 0.4;
  }
  .branch-arrow {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
  }
  .branch-arrow .ba-line { width: 2px; height: 24px; background: linear-gradient(to bottom, var(--blue), rgba(37,99,235,0.3)); }
  .branch-arrow .ba-arrow { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 8px solid rgba(37,99,235,0.7); }
  .branch-arrow .ba-label {
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    margin-top: 4px;
    text-align: center;
    padding: 2px 6px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  /* Merge indicator */
  .flow-merge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 0;
    width: 100%;
    max-width: 820px;
    color: var(--muted);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }
  .flow-merge::before,
  .flow-merge::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border));
  }
  .flow-merge::after { transform: scaleX(-1); }

  /* ── GitHub commit timeline ── */
  .commit-timeline { display: flex; flex-direction: column; gap: 0; }
  .commit-week {
    display: flex;
    gap: 0;
    align-items: stretch;
  }
  .week-label {
    width: 100px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    padding-right: 20px;
    padding-top: 14px;
  }
  .week-label .wl-num {
    font-size: 11px;
    font-weight: 700;
    color: var(--bblue);
    font-family: 'JetBrains Mono', monospace;
  }
  .week-label .wl-name {
    font-size: 10px;
    color: var(--muted);
    text-align: right;
    margin-top: 2px;
    line-height: 1.3;
  }
  .week-track {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 24px;
    flex-shrink: 0;
  }
  .wt-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 16px;
    z-index: 1;
    box-shadow: 0 0 10px currentColor;
  }
  .wt-line {
    flex: 1;
    width: 2px;
    background: var(--border);
    margin: 2px 0;
  }
  .commit-week:last-child .wt-line { display: none; }
  .week-commits {
    flex: 1;
    padding: 10px 0 10px 16px;
  }
  .week-commits .wc-header {
    font-size: 13px;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 8px;
  }
  .commit-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 8px;
    border-radius: 6px;
    margin-bottom: 3px;
    transition: background 0.15s;
  }
  .commit-row:hover { background: rgba(255,255,255,0.03); }
  .cr-day {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--muted);
    width: 28px;
    flex-shrink: 0;
  }
  .cr-type {
    font-size: 9px;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 10px;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
  }
  .cr-msg { font-size: 12px; color: var(--text); }
  .cr-file {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--muted);
    flex-shrink: 0;
  }
  .ct-feat   { background: rgba(22,163,74,0.15);  color: var(--lgreen);  border:1px solid rgba(22,163,74,0.3); }
  .ct-data   { background: rgba(13,148,136,0.15); color: var(--lteal);   border:1px solid rgba(13,148,136,0.3); }
  .ct-model  { background: rgba(124,58,237,0.15); color: var(--lpurple); border:1px solid rgba(124,58,237,0.3); }
  .ct-viz    { background: rgba(37,99,235,0.15);  color: var(--bblue);   border:1px solid rgba(37,99,235,0.3); }
  .ct-chore  { background: rgba(100,116,139,0.15);color: var(--muted);   border:1px solid rgba(100,116,139,0.3); }
  .ct-docs   { background: rgba(217,119,6,0.15);  color: var(--lamber);  border:1px solid rgba(217,119,6,0.3); }

  /* Responsive */
  @media(max-width:768px){
    .cards-row.cols-3,.cards-row.cols-4,.cards-row.cols-5{grid-template-columns:1fr 1fr;}
    .two-col{grid-template-columns:1fr;}
    .flow-step-body{grid-template-columns:1fr;}
    .flow-divider{display:none;}
    .pipe-steps{flex-direction:column;}
    .pipe-step:not(:last-child)::after{display:none;}
    .pipe-step+.pipe-step{margin-left:0;margin-top:12px;}
    .week-label{display:none;}
  }

  /* print-friendly separator */
  .page-sep { height: 1px; background: var(--border); margin: 60px 0; }

  /* glow accents */
  .glow-ring { filter: drop-shadow(0 0 12px rgba(59,130,246,0.4)); }
</style>
</head>
<body>
<div class="page-wrap">

  <!-- ════════════════════════════════════════════════════
       PAGE HEADER
  ═══════════════════════════════════════════════════════ -->
  <header class="page-header">
    <div class="badge">Portfolio Project · 2025</div>
    <h1>E-Commerce Predictive<br>Intelligence Platform</h1>
    <p>System Architecture &amp; Data Flow Diagrams — from raw CSV files to live predictions</p>
  </header>

  <!-- ════════════════════════════════════════════════════
       DIAGRAM 1 — SYSTEM ARCHITECTURE
  ═══════════════════════════════════════════════════════ -->
  <div class="section-label">
    <div class="num">1</div>
    <div>
      <h2>System Architecture Diagram</h2>
      <p>Every layer of the platform and how they connect</p>
    </div>
  </div>

  <div class="diagram-canvas">

    <!-- LAYER 1: DATA SOURCE -->
    <div class="layer layer-data">
      <div class="layer-header">
        <span class="dot"></span>
        LAYER 1 — DATA SOURCE
      </div>
      <div class="layer-body">
        <div class="cards-row cols-1" style="max-width:480px;margin:0 auto;">
          <div class="dataset-card">
            <div class="ds-title">📦 Olist Brazilian E-Commerce Dataset</div>
            <div class="ds-meta">
              Source: kaggle.com/datasets/olistbr/brazilian-ecommerce<br>
              Size: 100,000+ orders &nbsp;|&nbsp; 9 CSV files &nbsp;|&nbsp; ~130 MB<br>
              Period: September 2016 → October 2018
            </div>
            <div>
              <span class="ds-tag">olist_orders_dataset.csv</span>
              <span class="ds-tag">olist_order_items_dataset.csv</span>
              <span class="ds-tag">olist_customers_dataset.csv</span>
              <span class="ds-tag">olist_products_dataset.csv</span>
              <span class="ds-tag">olist_order_reviews_dataset.csv</span>
              <span class="ds-tag">olist_order_payments_dataset.csv</span>
              <span class="ds-tag">olist_sellers_dataset.csv</span>
              <span class="ds-tag">olist_geolocation_dataset.csv</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="connector"><div class="arrow-line"></div><div class="label">8 raw CSV files loaded into memory</div><div class="arrow-head"></div></div>

    <!-- LAYER 2: DATA PIPELINE -->
    <div class="layer layer-pipe">
      <div class="layer-header">
        <span class="dot"></span>
        LAYER 2 — DATA PIPELINE
      </div>
      <div class="layer-body">
        <div class="pipe-steps">
          <div class="pipe-step">
            <div class="step-file">data_loader.py</div>
            <div class="step-title">Load &amp; Validate</div>
            <div class="step-desc">Reads all 8 CSV files, checks column schemas, logs memory usage</div>
          </div>
          <div class="pipe-step">
            <div class="step-file">data_cleaner.py</div>
            <div class="step-title">Clean &amp; Fix</div>
            <div class="step-desc">Remove duplicates, parse dates, cap outliers, fix impossible values</div>
          </div>
          <div class="pipe-step">
            <div class="step-file">feature_engineer.py</div>
            <div class="step-title">Build Features</div>
            <div class="step-desc">Merge 8 tables, compute RFM, create 19 ML features, add churn label</div>
          </div>
        </div>
        <div style="margin-top:12px; padding:10px 14px; background:rgba(37,99,235,0.07); border-radius:8px; border:1px solid rgba(37,99,235,0.2);">
          <span style="font-size:11px; color:var(--muted);">Output → </span>
          <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--bblue);">data/processed/master_features.parquet</span>
          <span style="font-size:11px; color:var(--muted);"> · 99,441 rows × 19 features · 1 row per customer</span>
        </div>
      </div>
    </div>

    <div class="connector"><div class="arrow-line"></div><div class="label">master_features.parquet feeds all 6 models</div><div class="arrow-head"></div></div>

    <!-- LAYER 3: ML MODELS -->
    <div class="layer layer-ml">
      <div class="layer-header">
        <span class="dot"></span>
        LAYER 3 — ML MODEL LAYER · 6 Models
      </div>
      <div class="layer-body">
        <div class="cards-row cols-3" style="margin-bottom:12px;">
          <div class="arch-card">
            <div class="card-icon">🎯</div>
            <div class="card-title">Churn Prediction</div>
            <div class="card-algo">churn_model.py<br>XGBoost + SMOTE + Optuna</div>
            <span class="card-metric m-green">AUC 0.891</span>
            <span class="card-metric m-blue" style="margin-left:4px;">F1 0.807</span>
          </div>
          <div class="arch-card">
            <div class="card-icon">📦</div>
            <div class="card-title">Demand Forecasting</div>
            <div class="card-algo">forecasting_model.py<br>Prophet + LSTM Hybrid</div>
            <span class="card-metric m-teal">MAPE 5.9%</span>
          </div>
          <div class="arch-card">
            <div class="card-icon">👥</div>
            <div class="card-title">Segmentation</div>
            <div class="card-algo">segmentation_model.py<br>K-Means (k=4)</div>
            <span class="card-metric m-purple">Silhouette 0.42</span>
          </div>
        </div>
        <div class="cards-row cols-3">
          <div class="arch-card">
            <div class="card-icon">💰</div>
            <div class="card-title">CLV Prediction</div>
            <div class="card-algo">clv_model.py<br>LightGBM + Optuna</div>
            <span class="card-metric m-amber">R² 0.847</span>
          </div>
          <div class="arch-card">
            <div class="card-icon">🚨</div>
            <div class="card-title">Anomaly Detection</div>
            <div class="card-algo">anomaly_detector.py<br>Isolation Forest + Z-Score</div>
            <span class="card-metric m-rose">F1 0.870</span>
          </div>
          <div class="arch-card">
            <div class="card-icon">💬</div>
            <div class="card-title">Sentiment Analysis</div>
            <div class="card-algo">sentiment_analyzer.py<br>TextBlob + NLTK</div>
            <span class="card-metric m-blue">9 Features</span>
          </div>
        </div>
      </div>
    </div>

    <div class="connector"><div class="arrow-line"></div><div class="label">trained .pkl models saved to models/saved/</div><div class="arrow-head"></div></div>

    <!-- LAYER 4: MLOPS + OUTPUT -->
    <div class="two-col">
      <div class="layer layer-mlops">
        <div class="layer-header">
          <span class="dot"></span>
          LAYER 4a — MLOPS
        </div>
        <div class="layer-body">
          <div class="arch-card" style="margin-bottom:10px;">
            <div class="card-title">📊 MLflow Experiment Tracking</div>
            <div class="card-algo" style="margin-top:6px;">mlflow_tracker.py<br>Logs params + metrics + model files<br>Compare all training runs</div>
          </div>
          <div class="arch-card">
            <div class="card-title">🔍 Evidently Drift Monitor</div>
            <div class="card-algo" style="margin-top:6px;">drift_detector.py<br>Weekly KS-test on 9 features<br>Auto-alert if p-value &lt; 0.15</div>
          </div>
        </div>
      </div>
      <div class="layer layer-output">
        <div class="layer-header">
          <span class="dot"></span>
          LAYER 4b — OUTPUTS
        </div>
        <div class="layer-body">
          <div class="arch-card" style="margin-bottom:10px;">
            <div class="card-title">⚡ FastAPI REST Server</div>
            <div class="card-algo" style="margin-top:6px;">api/main.py<br>POST /predict/churn<br>&lt; 100ms response · Pydantic validation</div>
          </div>
          <div class="arch-card">
            <div class="card-title">📈 Interactive Dashboards</div>
            <div class="card-algo" style="margin-top:6px;">HTML/CSS/Bootstrap + Plotly<br>Power BI .pbix file<br>Open in any browser</div>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /diagram-canvas -->

  <div class="page-sep"></div>

  <!-- ════════════════════════════════════════════════════
       DIAGRAM 2 — DATA FLOW
  ═══════════════════════════════════════════════════════ -->
  <div class="section-label">
    <div class="num">2</div>
    <div>
      <h2>Data Flow Diagram</h2>
      <p>Step-by-step: how data moves from raw CSV to live API prediction</p>
    </div>
  </div>

  <div class="diagram-canvas">
  <div class="flow-wrap">

    <!-- STEP 1 -->
    <div class="flow-step">
      <div class="flow-step-header">
        <div class="flow-num" style="background:rgba(13,148,136,0.2);color:var(--lteal);border:1px solid rgba(13,148,136,0.4);">1</div>
        <div class="flow-title">Data Loading</div>
        <div class="flow-file">data_loader.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>8 CSV files in data/raw/</li>
            <li>~130 MB total disk size</li>
            <li>100,000+ order rows</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions</div>
          <ul>
            <li>Validate column schemas against SCHEMAS dict</li>
            <li>Log memory usage of each DataFrame</li>
            <li>Warn if expected columns are missing</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div><span class="out-tag">dict of 8 DataFrames</span><span class="out-tag">in memory</span></div>
          <div style="font-size:11px; color:var(--muted); margin-top:6px; font-family:'JetBrains Mono',monospace;">keys: orders, items, customers,<br>products, reviews, payments, sellers</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- STEP 2 -->
    <div class="flow-step">
      <div class="flow-step-header">
        <div class="flow-num" style="background:rgba(37,99,235,0.2);color:var(--bblue);border:1px solid rgba(37,99,235,0.4);">2</div>
        <div class="flow-title">Data Cleaning</div>
        <div class="flow-file">data_cleaner.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>8 raw DataFrames from Step 1</li>
            <li>Contains missing values</li>
            <li>Contains outliers &amp; errors</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions</div>
          <ul>
            <li>Drop duplicate order_ids</li>
            <li>Parse all date columns to datetime</li>
            <li>Remove delivery before purchase</li>
            <li>Cap price outliers (IQR method)</li>
            <li>Fill missing review text with ""</li>
            <li>Fill missing product weights with median</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div><span class="out-tag">8 clean DataFrames</span><span class="out-tag">cleaning report</span></div>
          <div style="font-size:11px; color:var(--muted); margin-top:6px;">~2-3% rows removed<br>All dates as datetime type<br>Prices capped at Q3+3×IQR</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- STEP 3 -->
    <div class="flow-step">
      <div class="flow-step-header">
        <div class="flow-num" style="background:rgba(37,99,235,0.2);color:var(--bblue);border:1px solid rgba(37,99,235,0.4);">3</div>
        <div class="flow-title">Feature Engineering</div>
        <div class="flow-file">feature_engineer.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>8 clean DataFrames</li>
            <li>Multiple rows per customer</li>
            <li>No ML-ready features yet</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions</div>
          <ul>
            <li>Merge all 8 tables on order_id / customer_id</li>
            <li>Add temporal features (hour, weekday, quarter)</li>
            <li>Calculate delivery delay per order</li>
            <li>Aggregate to 1 row per customer</li>
            <li>Compute RFM scores (r_score, f_score, m_score)</li>
            <li>Create churn label (recency &gt; 90 days)</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div><span class="out-tag">master_features.parquet</span></div>
          <div style="font-size:11px; color:var(--muted); margin-top:6px;">99,441 rows × 19 columns<br>1 row per unique customer<br>18% churn rate label</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- STEP 4 split label -->
    <div class="flow-merge">master_features.parquet feeds into all 6 models in parallel</div>

    <div class="flow-branch">
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">segmentation_model.py</div></div>
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">churn_model.py</div></div>
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">forecasting_model.py</div></div>
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">clv_model.py</div></div>
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">anomaly_detector.py</div></div>
      <div class="branch-arrow"><div class="ba-line"></div><div class="ba-arrow"></div><div class="ba-label">sentiment_analyzer.py</div></div>
    </div>

    <!-- STEP 4 ML block -->
    <div class="flow-step" style="border-color:rgba(124,58,237,0.4);">
      <div class="flow-step-header" style="background:rgba(124,58,237,0.07);">
        <div class="flow-num" style="background:rgba(124,58,237,0.2);color:var(--lpurple);border:1px solid rgba(124,58,237,0.4);">4</div>
        <div class="flow-title">Model Training (all 6 models)</div>
        <div class="flow-file">models/*.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>master_features.parquet</li>
            <li>19 feature columns</li>
            <li>churn label column</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions per model</div>
          <ul>
            <li>Scale features with StandardScaler</li>
            <li>Apply SMOTE for churn imbalance</li>
            <li>Run Optuna hyperparameter search</li>
            <li>Train final model with best params</li>
            <li>Evaluate on holdout test set</li>
            <li>Log metrics to MLflow</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div>
            <span class="out-tag">churn_model.pkl</span>
            <span class="out-tag">segmenter.pkl</span>
            <span class="out-tag">forecaster.pkl</span>
            <span class="out-tag">clv_model.pkl</span>
            <span class="out-tag">anomaly_detector.pkl</span>
            <span class="out-tag">sentiment scores</span>
          </div>
          <div style="font-size:11px; color:var(--muted); margin-top:6px;">All saved to models/saved/</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- STEP 5 -->
    <div class="flow-step" style="border-color:rgba(217,119,6,0.4);">
      <div class="flow-step-header" style="background:rgba(217,119,6,0.07);">
        <div class="flow-num" style="background:rgba(217,119,6,0.2);color:var(--lamber);border:1px solid rgba(217,119,6,0.4);">5</div>
        <div class="flow-title">Drift Monitoring</div>
        <div class="flow-file">drift_detector.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>Current production data</li>
            <li>reference_data.parquet (saved at training time)</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions</div>
          <ul>
            <li>KS-test each feature distribution</li>
            <li>Compare vs reference distributions</li>
            <li>Evidently generates HTML report</li>
            <li>Alert if any p-value &lt; 0.15</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div><span class="out-tag">drift_report_YYYYMMDD.html</span></div>
          <div style="font-size:11px; color:var(--muted); margin-top:6px;">Saved to reports/drift_reports/<br>Critical log alert if drift found</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- STEP 6 -->
    <div class="flow-step" style="border-color:rgba(22,163,74,0.4);">
      <div class="flow-step-header" style="background:rgba(22,163,74,0.07);">
        <div class="flow-num" style="background:rgba(22,163,74,0.2);color:var(--lgreen);border:1px solid rgba(22,163,74,0.4);">6</div>
        <div class="flow-title">Live Prediction Serving</div>
        <div class="flow-file">api/main.py</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Input</div>
          <ul>
            <li>HTTP POST with customer JSON</li>
            <li>recency_days, frequency, monetary…</li>
            <li>Validated by Pydantic schema</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Actions</div>
          <ul>
            <li>Load churn_model.pkl from disk</li>
            <li>Scale input features</li>
            <li>Predict probability with .predict_proba()</li>
            <li>Apply optimal threshold</li>
            <li>Map probability to risk level + action</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Output</div>
          <div><span class="out-tag">JSON response</span></div>
          <div style="font-size:11px; color:var(--muted); font-family:'JetBrains Mono',monospace; margin-top:6px; line-height:1.7;">churn_probability: 0.847<br>risk_level: "HIGH"<br>recommended_action: "…"<br>response_time: &lt;100ms</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow"><div class="fa-line"></div><div class="fa-arrow"></div></div>

    <!-- FINAL OUTPUT -->
    <div class="flow-step" style="border-color:rgba(22,163,74,0.5); box-shadow:0 0 0 1px rgba(22,163,74,0.2), 0 8px 32px rgba(22,163,74,0.1);">
      <div class="flow-step-header" style="background:rgba(22,163,74,0.1);">
        <div class="flow-num" style="background:var(--green);color:white;border:1px solid var(--lgreen);">✓</div>
        <div class="flow-title" style="color:var(--lgreen);">Final Deliverables</div>
        <div class="flow-file" style="color:var(--lgreen); border-color:rgba(22,163,74,0.35); background:rgba(22,163,74,0.1);">reports/ + api/</div>
      </div>
      <div class="flow-step-body">
        <div class="flow-col">
          <div class="flow-col-label">Dashboards</div>
          <ul>
            <li>HTML/Bootstrap dashboard (open in browser)</li>
            <li>Power BI .pbix file (4 report pages)</li>
            <li>EDA charts in reports/figures/</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">MLOps Artifacts</div>
          <ul>
            <li>MLflow UI at localhost:5000</li>
            <li>Evidently drift HTML reports</li>
            <li>Trained model .pkl files</li>
            <li>Pipeline log files in logs/</li>
          </ul>
        </div>
        <div class="flow-divider"></div>
        <div class="flow-col">
          <div class="flow-col-label">Live API</div>
          <ul>
            <li>FastAPI at localhost:8000</li>
            <li>Auto-docs at /docs</li>
            <li>Batch predictions endpoint</li>
            <li>Health check at /health</li>
          </ul>
        </div>
      </div>
    </div>

  </div><!-- /flow-wrap -->
  </div><!-- /diagram-canvas -->

  <div class="page-sep"></div>

  <!-- ════════════════════════════════════════════════════
       DIAGRAM 3 — GITHUB COMMIT TIMELINE
  ═══════════════════════════════════════════════════════ -->
  <div class="section-label">
    <div class="num">3</div>
    <div>
      <h2>GitHub Commit Sequence</h2>
      <p>Exactly which file to commit on each day — builds consistent green squares over 8 weeks</p>
    </div>
  </div>

  <div class="diagram-canvas">
  <div class="commit-timeline">

    <!-- WEEK 1 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 1</div><div class="wl-name">Project<br>Setup</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lteal);background:var(--lteal);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Project Setup & Data Loading</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-chore">chore</span><span class="cr-msg">initialize project folder structure and .gitignore</span><span class="cr-file">.gitignore</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-data">data</span><span class="cr-msg">add data_loader.py with schema validation for all 8 CSVs</span><span class="cr-file">data_loader.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-data">data</span><span class="cr-msg">add memory usage logging and error messages to loader</span><span class="cr-file">data_loader.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-chore">chore</span><span class="cr-msg">add requirements.txt with all project dependencies</span><span class="cr-file">requirements.txt</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-docs">docs</span><span class="cr-msg">add README with project overview and dataset links</span><span class="cr-file">README.md</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-data">data</span><span class="cr-msg">add column presence validation with detailed warnings</span><span class="cr-file">data_loader.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-chore">chore</span><span class="cr-msg">add GitHub Actions CI workflow for automated testing</span><span class="cr-file">.github/workflows/ci.yml</span></div>
      </div>
    </div>

    <!-- WEEK 2 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 2</div><div class="wl-name">Data<br>Cleaning</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--bblue);background:var(--bblue);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Data Cleaning</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-data">data</span><span class="cr-msg">add data_cleaner.py with orders cleaning and date parsing</span><span class="cr-file">data_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-data">data</span><span class="cr-msg">add IQR outlier capping for extreme price values</span><span class="cr-file">data_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-data">data</span><span class="cr-msg">add impossible delivery date removal business rule</span><span class="cr-file">data_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-data">data</span><span class="cr-msg">add cleaning methods for reviews, payments, products, customers</span><span class="cr-file">data_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add get_cleaning_report showing rows removed per dataset</span><span class="cr-file">data_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-data">data</span><span class="cr-msg">add temporal feature extraction to feature_engineer.py</span><span class="cr-file">feature_engineer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-docs">docs</span><span class="cr-msg">update README with data pipeline and cleaning steps</span><span class="cr-file">README.md</span></div>
      </div>
    </div>

    <!-- WEEK 3 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 3</div><div class="wl-name">Feature<br>Eng + EDA</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--bblue);background:var(--bblue);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Feature Engineering & EDA Notebooks</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-data">data</span><span class="cr-msg">add delivery feature engineering (delay_days, was_late)</span><span class="cr-file">feature_engineer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-data">data</span><span class="cr-msg">add customer-level aggregation to master feature table</span><span class="cr-file">feature_engineer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add full RFM scoring with quintile-based 1-5 scores</span><span class="cr-file">feature_engineer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add churn label engineering with 90-day threshold</span><span class="cr-file">feature_engineer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add EDA notebook with sales trend and geographic analysis</span><span class="cr-file">notebooks/01_eda.ipynb</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add review score distribution and delivery performance plots</span><span class="cr-file">notebooks/01_eda.ipynb</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add model_comparison.py testing 4 algorithms with cross-validation</span><span class="cr-file">model_comparison.py</span></div>
      </div>
    </div>

    <!-- WEEK 4 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 4</div><div class="wl-name">Segmen-<br>tation</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lpurple);background:var(--lpurple);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Customer Segmentation Model</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-model">model</span><span class="cr-msg">add CustomerSegmenter class with K-Means fit_predict</span><span class="cr-file">models/segmentation_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-model">model</span><span class="cr-msg">add elbow method and silhouette score for optimal k selection</span><span class="cr-file">models/segmentation_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add Champions / At-Risk / Loyalists / Bargain Hunter segment labels</span><span class="cr-file">models/segmentation_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-model">model</span><span class="cr-msg">add hierarchical agglomerative clustering for comparison</span><span class="cr-file">models/segmentation_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add get_segment_profile summary with business metrics</span><span class="cr-file">models/segmentation_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add RFM scatter plot and segment donut chart notebook</span><span class="cr-file">notebooks/02_segmentation.ipynb</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-docs">docs</span><span class="cr-msg">add segmentation results table and business interpretation to README</span><span class="cr-file">README.md</span></div>
      </div>
    </div>

    <!-- WEEK 5 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 5</div><div class="wl-name">Churn<br>+ CLV</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lgreen);background:var(--lgreen);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Churn Prediction & CLV Models</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-model">model</span><span class="cr-msg">add ChurnPredictor class with XGBoost baseline (AUC 0.841)</span><span class="cr-file">models/churn_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-model">model</span><span class="cr-msg">add SMOTE oversampling to handle 18% churn imbalance</span><span class="cr-file">models/churn_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-model">model</span><span class="cr-msg">add Optuna hyperparameter tuning — 50 trials (AUC → 0.891)</span><span class="cr-file">models/churn_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add SHAP explainability and threshold optimisation</span><span class="cr-file">models/churn_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-model">model</span><span class="cr-msg">add CLVPredictor with LightGBM and Optuna 40 trials</span><span class="cr-file">models/clv_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add CLV tier labels (Platinum / Gold / Silver / Bronze)</span><span class="cr-file">models/clv_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add ROC curve comparison and SHAP bar chart notebook</span><span class="cr-file">notebooks/03_churn.ipynb</span></div>
      </div>
    </div>

    <!-- WEEK 6 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 6</div><div class="wl-name">Forecast<br>+ NLP</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lteal);background:var(--lteal);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Forecasting, Anomaly Detection & Sentiment NLP</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-model">model</span><span class="cr-msg">add Prophet demand forecasting model with Brazilian holidays</span><span class="cr-file">models/forecasting_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-model">model</span><span class="cr-msg">add LSTM neural network for time series forecasting</span><span class="cr-file">models/forecasting_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-model">model</span><span class="cr-msg">add Prophet-LSTM hybrid ensemble achieving MAPE 5.9%</span><span class="cr-file">models/forecasting_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-model">model</span><span class="cr-msg">add AnomalyDetector with Isolation Forest and Z-Score layer</span><span class="cr-file">models/anomaly_detector.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add auto-explain reason for each flagged anomaly</span><span class="cr-file">models/anomaly_detector.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-model">model</span><span class="cr-msg">add SentimentAnalyzer with TextBlob — 9 text features</span><span class="cr-file">models/sentiment_analyzer.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add complaint keyword extraction and churn correlation analysis</span><span class="cr-file">models/sentiment_analyzer.py</span></div>
      </div>
    </div>

    <!-- WEEK 7 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 7</div><div class="wl-name">MLOps<br>+ API</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lamber);background:var(--lamber);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">MLOps, Monitoring & FastAPI</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add MLflow experiment tracking to churn and CLV models</span><span class="cr-file">models/churn_model.py · models/clv_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add MLflow tracking to forecasting and segmentation models</span><span class="cr-file">models/forecasting_model.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add drift_detector.py with KS-test and Evidently HTML reports</span><span class="cr-file">monitoring/drift_detector.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add weekly drift monitoring with auto-alert on p-value threshold</span><span class="cr-file">monitoring/drift_detector.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add FastAPI server with POST /predict/churn endpoint</span><span class="cr-file">api/main.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add batch prediction and /model/info endpoints to API</span><span class="cr-file">api/main.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-feat">feat</span><span class="cr-msg">add main_pipeline.py orchestrating all stages end-to-end</span><span class="cr-file">main_pipeline.py</span></div>
      </div>
    </div>

    <!-- WEEK 8 -->
    <div class="commit-week">
      <div class="week-label"><div class="wl-num">WEEK 8</div><div class="wl-name">Dashboards<br>+ Polish</div></div>
      <div class="week-track"><div class="wt-dot" style="color:var(--lgreen);background:var(--lgreen);"></div><div class="wt-line"></div></div>
      <div class="week-commits">
        <div class="wc-header">Dashboards, README Polish & v1.0.0 Release</div>
        <div class="commit-row"><span class="cr-day">Day 1</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add HTML dashboard with Bootstrap and Plotly interactive charts</span><span class="cr-file">reports/html/dashboard.html</span></div>
        <div class="commit-row"><span class="cr-day">Day 2</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add churn risk table and segment profile section to dashboard</span><span class="cr-file">reports/html/dashboard.html</span></div>
        <div class="commit-row"><span class="cr-day">Day 3</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add demand forecast chart and MLOps model registry section</span><span class="cr-file">reports/html/dashboard.html</span></div>
        <div class="commit-row"><span class="cr-day">Day 4</span><span class="cr-type ct-viz">viz</span><span class="cr-msg">add architecture and data flow HTML diagrams to project</span><span class="cr-file">reports/html/architecture_diagrams.html</span></div>
        <div class="commit-row"><span class="cr-day">Day 5</span><span class="cr-type ct-docs">docs</span><span class="cr-msg">add dashboard screenshots and model results table to README</span><span class="cr-file">README.md</span></div>
        <div class="commit-row"><span class="cr-day">Day 6</span><span class="cr-type ct-chore">chore</span><span class="cr-msg">add unit tests for data_cleaner and feature_engineer</span><span class="cr-file">tests/test_cleaner.py</span></div>
        <div class="commit-row"><span class="cr-day">Day 7</span><span class="cr-type ct-chore">chore</span><span class="cr-msg">release: v1.0.0 — complete ML platform · git tag -a v1.0.0</span><span class="cr-file">v1.0.0 tag</span></div>
      </div>
    </div>

  </div><!-- /commit-timeline -->
  </div><!-- /diagram-canvas -->

  <!-- footer -->
  <div style="text-align:center; margin-top:48px; color:var(--muted); font-size:12px;">
    E-Commerce Predictive Intelligence Platform &nbsp;·&nbsp; 56 commits over 8 weeks &nbsp;·&nbsp; 2025
  </div>

</div><!-- /page-wrap -->
</body>
</html>

<br/>

---

## 🔄 Data Flow Diagram

> How data moves through each file, step by step.

<br/>

```
  START
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — data_loader.py                                            │
│                                                                     │
│  Input  → data/raw/*.csv  (8 Olist CSV files on your computer)      │
│  Action → Validates column names  |  Logs memory usage              │
│  Output → Python dictionary with 8 DataFrames loaded in memory      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — data_cleaner.py                                           │
│                                                                     │
│  Input  → 8 raw DataFrames from Step 1                              │
│  Action → Remove duplicates  |  Fix dates  |  Cap price outliers    │
│           Remove impossible delivery dates  |  Fill missing values  │
│  Output → 8 clean DataFrames  +  cleaning report (rows removed)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — feature_engineer.py                                       │
│                                                                     │
│  Input  → 8 clean DataFrames from Step 2                            │
│  Action → Merge all 8 tables on order_id / customer_id              │
│           Create RFM scores  |  Delivery features  |  Churn label   │
│           Aggregate to ONE ROW PER CUSTOMER                         │
│  Output → master_features.parquet  (99,441 rows × 19 features)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  STEP 4a         │  │  STEP 4b         │  │  STEP 4c             │
│  segmentation    │  │  churn_model.py  │  │  forecasting_model   │
│  _model.py       │  │                  │  │  .py                 │
│                  │  │  SMOTE balance   │  │                      │
│  K-Means k=4     │  │  Optuna 50 trials│  │  Prophet model       │
│  Find clusters   │  │  XGBoost train   │  │  LSTM neural net     │
│  Label segments  │  │  SHAP explain    │  │  Weighted ensemble   │
│  Save .pkl       │  │  Save .pkl       │  │  Save .pkl           │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  STEP 4d         │  │  STEP 4e         │  │  STEP 4f             │
│  clv_model.py    │  │  anomaly_        │  │  sentiment_          │
│                  │  │  detector.py     │  │  analyzer.py         │
│  LightGBM train  │  │                  │  │                      │
│  Optuna 40 trials│  │  Isolation Forest│  │  TextBlob + NLTK     │
│  CLV tiers added │  │  Z-Score layer   │  │  9 text features     │
│  Save .pkl       │  │  Auto-explain    │  │  Polarity scores     │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
          │                    │                     │
          └────────────────────┴─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — monitoring/drift_detector.py                              │
│                                                                     │
│  Input  → Current production data  +  reference_data.parquet        │
│  Action → KS-test on all 9 features  |  Evidently HTML report       │
│           Alert if p-value < 0.15 on any feature                    │
│  Output → drift_report_YYYYMMDD.html  (saved to reports/drift/)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6 — api/main.py   (Run separately: uvicorn api.main:app)      │
│                                                                     │
│  Input  → HTTP POST request with customer JSON data                 │
│  Action → Load .pkl models  |  Validate request (Pydantic)          │
│           Scale features  |  Predict probability                    │
│           Apply threshold  |  Add business logic                    │
│  Output → JSON response with churn prob + risk level + action       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  reports/html/       │
                    │  dashboard.html      │
                    │                      │
                    │  Open in any browser │
                    │  No software needed  │
                    └──────────────────────┘
                            DONE ✓
```

<br/>

---

## 📦 Datasets

<div align="center">

| # | Dataset | Source | Size | Used For |
|:---:|:---|:---|:---:|:---|
| 1 | **Olist Brazilian E-Commerce** | [📥 Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | 100K+ rows | Primary — all 6 models |


</div>

<br/>

### Dataset 1 — Olist: File Structure

```
data/raw/
├── olist_orders_dataset.csv          ← Main orders table (order_id, status, timestamps)
├── olist_order_items_dataset.csv     ← Items in each order (product_id, price, freight)
├── olist_customers_dataset.csv       ← Customer info (unique_id, city, state)
├── olist_products_dataset.csv        ← Product details (category, weight, dimensions)
├── olist_order_reviews_dataset.csv   ← Customer reviews (score 1-5, comment text)
├── olist_order_payments_dataset.csv  ← Payment info (type, installments, value)
├── olist_sellers_dataset.csv         ← Seller location (city, state)
└── olist_geolocation_dataset.csv     ← ZIP code → GPS coordinates
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+    Git    8GB RAM minimum    10GB disk space
```

### 1 — Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-predictive-intelligence-platform.git
cd ecommerce-predictive-intelligence-platform

# Create isolated environment (always use virtual environments!)
python -m venv venv
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt   # Takes 5-10 minutes
```

### 2 — Download Data

```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json API key in ~/.kaggle/kaggle.json
# Then download the main Olist dataset:
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

# Verify files are present
ls data/raw/
# Should show 8 or 9 CSV files
```

### 3 — Run Full Pipeline

```bash
# Run everything end-to-end (first time: ~60 minutes due to model training)
python main_pipeline.py --stage all

# Or run individual stages:
python main_pipeline.py --stage data      # Data cleaning + feature engineering only
python main_pipeline.py --stage models    # Train all 6 ML models only
python main_pipeline.py --stage monitor   # Run drift detection only
```

### 4 — View Results

```bash
# Option A: Open HTML dashboard in your browser
open reports/html/dashboard.html          # Mac
start reports/html/dashboard.html         # Windows

# Option B: Launch MLflow experiment tracking UI
mlflow ui --port 5000
# Open http://localhost:5000 in browser

# Option C: Start the prediction API
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs for interactive API documentation
```

### 5 — Test the API

```bash
# Test churn prediction for a single customer
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "recency_days": 85,
    "frequency": 2,
    "monetary": 250.0,
    "avg_review_score": 3.2,
    "pct_late_deliveries": 0.5
  }'

# Expected response:
# {
#   "churn_probability": 0.847,
#   "churn_prediction": 1,
#   "risk_level": "HIGH",
#   "recommended_action": "Send immediate win-back offer (25% discount + free shipping)",
#   "model_version": "xgboost_v2.0"
# }
```

---

## 📊 Model Results

<div align="center">

### Classification Models

| Model | Algorithm | Metric | Score | Status |
|:---|:---|:---|:---:|:---:|
| **Churn Prediction** | XGBoost + SMOTE + Optuna | ROC-AUC | **0.891** | ✅ Production |
| **Churn Prediction** | XGBoost + SMOTE + Optuna | F1-Score | **0.807** | ✅ Production |
| **Churn Prediction** | XGBoost + SMOTE + Optuna | Precision | **0.834** | ✅ Production |
| **Anomaly Detection** | Isolation Forest | F1-Score | **0.870** | ✅ Production |

### Forecasting Model

| Model | Algorithm | Metric | Score | Target | Status |
|:---|:---|:---|:---:|:---:|:---:|
| **Demand Forecasting** | Prophet only | MAPE | 6.8% | < 7% | ✅ Pass |
| **Demand Forecasting** | LSTM only | MAPE | 7.1% | < 7% | ⚠️ Borderline |
| **Demand Forecasting** | **Hybrid Ensemble** | **MAPE** | **5.9%** | **< 7%** | ✅ **Best** |

### Regression & Clustering Models

| Model | Algorithm | Metric | Score | Status |
|:---|:---|:---|:---:|:---:|
| **CLV Prediction** | LightGBM + Optuna | RMSE | **82.4** | ✅ Production |
| **CLV Prediction** | LightGBM + Optuna | R² Score | **0.847** | ✅ Production |
| **Customer Segmentation** | K-Means (k=4) | Silhouette | **0.42** | ✅ Production |

</div>

---

## 👥 Customer Segments

<div align="center">

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend | Churn Rate | Action |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🏆 **Champions** | 12,408 | 18 days | 5.2 | R$621 | 3.1% | VIP treatment · Early access |
| 💙 **Potential Loyalists** | 28,935 | 41 days | 2.8 | R$248 | 11.4% | Loyalty points · Personalised reco |
| ⚠️ **At-Risk** | 19,887 | 82 days | 2.1 | R$174 | 41.8% | Win-back campaign · 25% discount |
| 🏷️ **Bargain Hunters** | 38,211 | 134 days | 1.3 | R$89 | 67.2% | Sale alerts only |

</div>

---

## 📁 Complete File Guide

```
ecommerce-predictive-intelligence-platform/
│
├── 📄 main_pipeline.py              ← START HERE — runs everything in correct order
├── 📄 data_loader.py                ← Step 1: Load & validate all 8 CSV files
├── 📄 data_cleaner.py               ← Step 2: Fix missing values, outliers, dates
├── 📄 feature_engineer.py           ← Step 3: Build RFM + 19 ML-ready features
├── 📄 requirements.txt              ← Install all libraries: pip install -r requirements.txt
│
├── 📁 models/
│   ├── 📄 segmentation_model.py     ← K-Means clustering → 4 customer segments
│   ├── 📄 churn_model.py            ← XGBoost churn predictor (AUC 0.891)
│   ├── 📄 forecasting_model.py      ← Prophet + LSTM demand forecast (MAPE 5.9%)
│   ├── 📄 clv_model.py              ← LightGBM CLV predictor (R² 0.847)
│   ├── 📄 anomaly_detector.py       ← Isolation Forest fraud detector
│   └── 📄 sentiment_analyzer.py     ← TextBlob review sentiment (9 features)
│
├── 📁 monitoring/
│   └── 📄 drift_detector.py         ← Weekly KS-test drift monitoring + alerts
│
├── 📁 api/
│   └── 📄 main.py                   ← FastAPI server: POST /predict/churn
│
├── 📁 data/
│   ├── 📁 raw/                      ← PUT YOUR CSV FILES HERE (gitignored)
│   ├── 📁 processed/                ← Auto-generated cleaned parquet files
│   └── 📁 sample/                   ← Small 1000-row sample files (in GitHub)
│
├── 📁 models/saved/                 ← Auto-generated trained model .pkl files
│
├── 📁 reports/
│   ├── 📁 html/                     ← Interactive HTML dashboard (open in browser)
│   ├── 📁 figures/                  ← EDA charts saved as PNG
│   └── 📁 drift_reports/            ← Weekly Evidently AI HTML drift reports
│
├── 📁 notebooks/                    ← 8 Jupyter notebooks (step-by-step analysis)
├── 📁 tests/                        ← Unit tests (pytest)
└── 📁 .github/workflows/            ← GitHub Actions CI pipeline
```

---

## 🛠️ Complete Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Language** | Python 3.11 | Core programming language |
| **Data** | Pandas · NumPy · SQLAlchemy | Data loading, cleaning, manipulation |
| **Visualisation** | Matplotlib · Seaborn · Plotly | Charts, EDA plots, interactive dashboard |
| **ML — Classical** | Scikit-learn · XGBoost · LightGBM | Churn, CLV, Segmentation, Anomaly |
| **ML — Deep Learning** | TensorFlow / Keras | LSTM neural network for forecasting |
| **ML — Forecasting** | Prophet (Meta) | Time series with seasonality |
| **ML — NLP** | TextBlob · NLTK | Sentiment analysis on reviews |
| **Optimisation** | Optuna | Automated hyperparameter tuning |
| **Explainability** | SHAP | Feature importance for model decisions |
| **Imbalance** | imbalanced-learn (SMOTE) | Handle 18% churn class imbalance |
| **MLOps — Tracking** | MLflow | Experiment logging and model registry |
| **MLOps — Monitoring** | Evidently AI | Data drift and model drift detection |
| **API** | FastAPI · Uvicorn · Pydantic | Real-time prediction serving |
| **Storage** | Parquet (PyArrow) | Efficient processed data storage |
| **Version Control** | Git · GitHub | Code management and portfolio |

</div>

---





## 📂 Output Files Reference

| File | Location | How to Open |
|:---|:---|:---|
| Interactive Dashboard | `reports/html/dashboard.html` | Double-click → opens in browser |
| MLflow Experiments | Auto-saved to `mlruns/` | Run `mlflow ui` → visit localhost:5000 |
| Drift Reports | `reports/drift_reports/*.html` | Double-click → opens in browser |
| Trained Models | `models/saved/*.pkl` | Load with `pickle.load()` in Python |
| Processed Data | `data/processed/*.parquet` | Load with `pd.read_parquet()` |
| Pipeline Logs | `logs/pipeline_YYYYMMDD.log` | Open in any text editor |
| API Documentation | Starts with FastAPI server | Visit `localhost:8000/docs` |

---



## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


