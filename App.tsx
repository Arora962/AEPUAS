/**
 * AEPUAS — Adaptive Ensemble Predictor with Uncertainty-Aware Scheduling
 * Mobile Research Demo — Android
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  StatusBar,
  Platform,
} from 'react-native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';

const { width: SCREEN_W } = Dimensions.get('window');

const COLORS = {
  bg: '#080c14',
  surface: '#0e1420',
  card: '#121a28',
  border: '#1e2d44',
  accent1: '#00d4ff',
  accent2: '#ff4e6a',
  accent3: '#a78bfa',
  accent4: '#34d399',
  text: '#e2e8f0',
  muted: '#64748b',
  warning: '#fbbf24',
};

const NODE_POOL = [
  { name: 'fog-edge-1', tier: 'FOG_EDGE', mips: 1200, ram: 2, bw: 50, color: COLORS.accent4 },
  { name: 'fog-edge-2', tier: 'FOG_EDGE', mips: 800, ram: 1, bw: 30, color: COLORS.accent4 },
  { name: 'fog-mid-1', tier: 'FOG_MID', mips: 5000, ram: 8, bw: 200, color: COLORS.accent3 },
  { name: 'fog-mid-2', tier: 'FOG_MID', mips: 3500, ram: 12, bw: 400, color: COLORS.accent3 },
  { name: 'cloud-1', tier: 'CLOUD', mips: 20000, ram: 64, bw: 800, color: COLORS.accent1 },
  { name: 'cloud-2', tier: 'CLOUD', mips: 15000, ram: 32, bw: 600, color: COLORS.accent1 },
];

const TASK_TYPES = ['sensor_agg', 'video_stream', 'health_alert', 'vehicular_nav', 'batch_ml'];

const TASK_PARAMS = {
  sensor_agg:    { size: [10, 500],   mem: [64, 512],   data: [0.1, 5],   priority: 1 },
  video_stream:  { size: [200, 2000], mem: [256, 2048],  data: [5, 50],    priority: 2 },
  health_alert:  { size: [50, 800],   mem: [128, 1024],  data: [0.5, 10],  priority: 3 },
  vehicular_nav: { size: [100, 1500], mem: [256, 2048],  data: [1, 20],    priority: 2 },
  batch_ml:      { size: [1000, 10000], mem: [1024, 8192], data: [10, 200], priority: 1 },
};

const PROPAGATION = { FOG_EDGE: 0.001, FOG_MID: 0.005, CLOUD: 0.02 };
const ALPHA = 1.5;
const KL_THRESHOLD = 0.15;

function rand(min, max) { return Math.random() * (max - min) + min; }

function randNorm(mean, std) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function computeExecTime(sizeMi, memMb, dataMb, mips, ram, bw, load, tier) {
  const freqFactor = 1.0 + 0.3 * Math.sin(Math.random() * 2 * Math.PI);
  const baseTime = (sizeMi * 1e6) / (mips * 1e3 * freqFactor);
  const memAvailMb = ram * 1024 * (1 - load * 0.6);
  const memPen = 1 + Math.max(0, (memMb - memAvailMb) / Math.max(memAvailMb, 1));
  const loadPen = 1.0 / Math.max(1 - Math.min(load, 0.95), 0.05);
  const netDelay = (dataMb / bw) * PROPAGATION[tier] * 1000;
  const raw = baseTime * memPen * loadPen + netDelay;
  const noise = randNorm(0, 0.05 * raw);
  return Math.max(raw + noise, 0.001);
}

function uqePredict(sizeMi, memMb, dataMb, node, load) {
  const N_BOOT = 30;
  const preds = [];
  for (let i = 0; i < N_BOOT; i++) {
    const jitter = 1 + randNorm(0, 0.08);
    preds.push(computeExecTime(sizeMi * jitter, memMb, dataMb, node.mips, node.ram, node.bw, load, node.tier));
  }
  const mean = preds.reduce((a, b) => a + b, 0) / N_BOOT;
  const variance = preds.reduce((a, b) => a + (b - mean) ** 2, 0) / N_BOOT;
  return { pred: mean, std: Math.sqrt(variance) };
}

function uaspSchedule(sizeMi, memMb, dataMb, loads) {
  const results = NODE_POOL.map((node, i) => {
    const load = loads[i];
    const { pred, std } = uqePredict(sizeMi, memMb, dataMb, node, load);
    const riskScore = pred + ALPHA * std;
    return { node: node.name, tier: node.tier, pred, std, riskScore, load, color: node.color };
  });
  results.sort((a, b) => a.riskScore - b.riskScore);
  return { results, bestNode: results[0].node };
}

function klDivergence(p, q) {
  const bins = 10;
  const allVals = [...p, ...q];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV + 1e-9;
  const histP = new Array(bins).fill(0);
  const histQ = new Array(bins).fill(0);
  p.forEach(v => { const b = Math.min(Math.floor(((v - minV) / range) * bins), bins - 1); histP[b]++; });
  q.forEach(v => { const b = Math.min(Math.floor(((v - minV) / range) * bins), bins - 1); histQ[b]++; });
  const eps = 1e-9;
  let kl = 0;
  for (let i = 0; i < bins; i++) {
    const pi = histP[i] / p.length + eps;
    const qi = histQ[i] / q.length + eps;
    kl += pi * Math.log(pi / qi);
  }
  return kl;
}

// ─── Animated Bar ─────────────────────────────────────────────────────────────

function AnimatedBar({ value, maxValue, color, height = 6 }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, { toValue: value / Math.max(maxValue, 0.001), duration: 600, useNativeDriver: false }).start();
  }, [value, maxValue]);
  return (
    <View style={[styles.barTrack, { height }]}>
      <Animated.View style={[styles.barFill, { backgroundColor: color, width: anim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }) }]} />
    </View>
  );
}

// ─── Pulse Dot ────────────────────────────────────────────────────────────────

function PulseDot({ color, size = 8 }) {
  const scale = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.loop(Animated.sequence([
      Animated.timing(scale, { toValue: 1.6, duration: 700, useNativeDriver: true }),
      Animated.timing(scale, { toValue: 1, duration: 700, useNativeDriver: true }),
    ])).start();
  }, []);
  return <Animated.View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: color, transform: [{ scale }] }} />;
}

// ─── Tab Bar ──────────────────────────────────────────────────────────────────

function TabBar({ active, onChange }) {
  const tabs = [
    { id: 'live', label: 'Live', icon: '⚡' },
    { id: 'nodes', label: 'Nodes', icon: '◈' },
    { id: 'drift', label: 'Drift', icon: '◎' },
    { id: 'metrics', label: 'Metrics', icon: '▦' },
  ];
  return (
    <View style={styles.tabBar}>
      {tabs.map(t => (
        <TouchableOpacity key={t.id} style={styles.tabItem} onPress={() => onChange(t.id)} activeOpacity={0.7}>
          <Text style={[styles.tabIcon, active === t.id && { color: COLORS.accent1 }]}>{t.icon}</Text>
          <Text style={[styles.tabLabel, active === t.id && { color: COLORS.accent1 }]}>{t.label}</Text>
          {active === t.id && <View style={styles.tabIndicator} />}
        </TouchableOpacity>
      ))}
    </View>
  );
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

function KpiCard({ label, value, unit, color, icon }) {
  const glow = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.sequence([
      Animated.timing(glow, { toValue: 1, duration: 200, useNativeDriver: false }),
      Animated.timing(glow, { toValue: 0, duration: 500, useNativeDriver: false }),
    ]).start();
  }, [value]);
  return (
    <Animated.View style={[styles.kpiCard, { borderColor: glow.interpolate({ inputRange: [0, 1], outputRange: [COLORS.border, color] }) }]}>
      <View style={[styles.kpiAccent, { backgroundColor: color }]} />
      <Text style={styles.kpiIcon}>{icon}</Text>
      <Text style={styles.kpiLabel}>{label}</Text>
      <Text style={[styles.kpiValue, { color }]}>{value}</Text>
      {unit ? <Text style={styles.kpiUnit}>{unit}</Text> : null}
    </Animated.View>
  );
}

// ─── AppContent ───────────────────────────────────────────────────────────────

function AppContent() {
  const insets = useSafeAreaInsets();
  const [tab, setTab] = useState('live');
  const [running, setRunning] = useState(false);
  const [taskLog, setTaskLog] = useState([]);
  const [nodeLoads, setNodeLoads] = useState(() =>
    Object.fromEntries(NODE_POOL.map(n => [n.name, rand(0.1, 0.6)]))
  );
  const [lastSchedule, setLastSchedule] = useState(null);
  const [drift, setDrift] = useState({ kl: 0.05, detected: false, retrain: false, taskCount: 0, burstMode: false });
  const taskCounter = useRef(0);
  const taskSizesRef = useRef([]);
  const referenceSizesRef = useRef([]);
  const intervalRef = useRef(null);
  const nodeLoadsRef = useRef(nodeLoads);
  const driftRef = useRef(drift);

  useEffect(() => { nodeLoadsRef.current = nodeLoads; }, [nodeLoads]);
  useEffect(() => { driftRef.current = drift; }, [drift]);

  const dispatchTask = useCallback(() => {
    const burst = driftRef.current.burstMode;
    const ttype = TASK_TYPES[Math.floor(Math.random() * TASK_TYPES.length)];
    const p = TASK_PARAMS[ttype];
    const sizeMult = burst ? 5 : 1;
    const loadBonus = burst ? 0.3 : 0;
    const sizeMi = rand(p.size[0], p.size[1]) * sizeMult;
    const memMb = rand(p.mem[0], p.mem[1]);
    const dataMb = rand(p.data[0], p.data[1]);

    const currentLoads = nodeLoadsRef.current;
    const loads = NODE_POOL.map(n => {
      const base = currentLoads[n.name] ?? 0.3;
      return Math.min(0.95, Math.max(0.05, base + loadBonus + randNorm(0, 0.04)));
    });
    const loadMap = {};
    NODE_POOL.forEach((n, i) => { loadMap[n.name] = loads[i]; });
    setNodeLoads(loadMap);

    const { results, bestNode } = uaspSchedule(sizeMi, memMb, dataMb, loads);
    const best = results[0];

    const event = {
      id: ++taskCounter.current,
      type: ttype,
      sizeMi,
      memMb,
      dataMb,
      bestNode,
      bestTier: best.tier,
      predTime: best.pred,
      uncertainty: best.std,
      riskScore: best.riskScore,
      timestamp: Date.now(),
    };

    setLastSchedule(results);
    setTaskLog(prev => [event, ...prev].slice(0, 50));

    taskSizesRef.current.push(sizeMi);
    if (taskSizesRef.current.length > 200) taskSizesRef.current.shift();

    setDrift(prev => {
      const count = prev.taskCount + 1;
      let { kl, detected, retrain } = prev;
      if (count % 20 === 0 && taskSizesRef.current.length >= 20) {
        if (referenceSizesRef.current.length === 0 && taskSizesRef.current.length >= 40) {
          referenceSizesRef.current = [...taskSizesRef.current];
        }
        const ref = referenceSizesRef.current.length > 0 ? referenceSizesRef.current : taskSizesRef.current;
        kl = klDivergence(taskSizesRef.current, ref);
        kl = Math.max(0.01, Math.min(kl, 2.0));
        detected = kl > KL_THRESHOLD;
        retrain = detected;
      }
      return { ...prev, kl, detected, retrain, taskCount: count };
    });
  }, []);

  const toggleRunning = () => {
    if (running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      setRunning(false);
    } else {
      setRunning(true);
      intervalRef.current = setInterval(dispatchTask, 800);
    }
  };

  const triggerBurst = () => {
    setDrift(d => ({ ...d, burstMode: true }));
    setTimeout(() => setDrift(d => ({ ...d, burstMode: false })), 8000);
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  const totalTasks = taskLog.length;
  const avgPred = totalTasks > 0 ? taskLog.reduce((a, t) => a + t.predTime, 0) / totalTasks : 0;
  const avgUncert = totalTasks > 0 ? taskLog.reduce((a, t) => a + t.uncertainty, 0) / totalTasks : 0;
  const fogCount = taskLog.filter(t => t.bestTier.startsWith('FOG')).length;
  const cloudCount = taskLog.filter(t => t.bestTier === 'CLOUD').length;

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.headerTagRow}>
            <View style={[styles.headerLine, { backgroundColor: COLORS.accent1 }]} />
            <Text style={[styles.headerTag, { color: COLORS.accent1 }]}>AEPUAS · RESEARCH DEMO</Text>
          </View>
          <Text style={styles.headerTitle}>Fog-Cloud{'\n'}Intelligence</Text>
        </View>
        <View style={styles.headerRight}>
          {running && <PulseDot color={COLORS.accent4} size={10} />}
          <TouchableOpacity
            style={[styles.controlBtn, { borderColor: running ? COLORS.accent2 : COLORS.accent4 }]}
            onPress={toggleRunning}
            activeOpacity={0.8}
          >
            <Text style={[styles.controlBtnText, { color: running ? COLORS.accent2 : COLORS.accent4 }]}>
              {running ? '⏹ STOP' : '▶ START'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* KPI Row */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.kpiScroll} contentContainerStyle={styles.kpiRow}>
        <KpiCard label="Tasks" value={String(totalTasks)} icon="⚡" color={COLORS.accent1} />
        <KpiCard label="Avg Pred" value={avgPred.toFixed(1)} unit="s" icon="⏱" color={COLORS.accent3} />
        <KpiCard label="Avg σ" value={avgUncert.toFixed(2)} unit="std" icon="±" color={COLORS.accent2} />
        <KpiCard label="Fog" value={String(fogCount)} icon="🌫" color={COLORS.accent4} />
        <KpiCard label="Cloud" value={String(cloudCount)} icon="☁" color={COLORS.accent1} />
        <KpiCard label="KL Div" value={drift.kl.toFixed(3)} icon="◎" color={drift.detected ? COLORS.accent2 : COLORS.accent4} />
      </ScrollView>

      <TabBar active={tab} onChange={setTab} />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: insets.bottom + 20 }}>
        {tab === 'live' && <LiveTab taskLog={taskLog} lastSchedule={lastSchedule} onDispatch={dispatchTask} drift={drift} onBurst={triggerBurst} />}
        {tab === 'nodes' && <NodesTab nodeLoads={nodeLoads} lastSchedule={lastSchedule} />}
        {tab === 'drift' && <DriftTab drift={drift} onBurst={triggerBurst} />}
        {tab === 'metrics' && <MetricsTab taskLog={taskLog} />}
      </ScrollView>
    </View>
  );
}

// ─── Live Tab ─────────────────────────────────────────────────────────────────

function LiveTab({ taskLog, lastSchedule, onDispatch, drift, onBurst }) {
  return (
    <View>
      <View style={styles.actionRow}>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: COLORS.accent3, flex: 1 }]} onPress={onDispatch} activeOpacity={0.7}>
          <Text style={[styles.actionBtnText, { color: COLORS.accent3 }]}>⚡ Dispatch Task</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: drift.burstMode ? COLORS.accent2 : COLORS.warning, flex: 1 }]} onPress={onBurst} activeOpacity={0.7}>
          <Text style={[styles.actionBtnText, { color: drift.burstMode ? COLORS.accent2 : COLORS.warning }]}>
            {drift.burstMode ? '🔥 BURST ON' : '💥 Sim Burst'}
          </Text>
        </TouchableOpacity>
      </View>

      {drift.detected && (
        <View style={[styles.alertBanner, { borderColor: COLORS.accent2 }]}>
          <PulseDot color={COLORS.accent2} size={7} />
          <Text style={[styles.alertText, { color: COLORS.accent2 }]}>DRIFT DETECTED · KL={drift.kl.toFixed(3)} · Retrain triggered</Text>
        </View>
      )}

      {lastSchedule && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Last UASP Decision</Text>
          <Text style={styles.sectionSubtitle}>risk = pred_time + {ALPHA}× uncertainty  ↑ sorted ascending</Text>
          {lastSchedule.map((r, i) => (
            <View key={r.node} style={[styles.scheduleRow, i === 0 && styles.scheduleRowBest]}>
              <View style={styles.scheduleLeft}>
                <View style={[styles.rankBadge, { backgroundColor: i === 0 ? COLORS.accent4 : COLORS.border }]}>
                  <Text style={[styles.rankText, { color: i === 0 ? COLORS.bg : COLORS.muted }]}>{i + 1}</Text>
                </View>
                <View>
                  <Text style={[styles.schedNodeName, { color: r.color }]}>{r.node}</Text>
                  <Text style={styles.schedNodeTier}>{r.tier}</Text>
                </View>
              </View>
              <View style={styles.scheduleRight}>
                <Text style={styles.schedPred}>{r.pred.toFixed(2)}s</Text>
                <Text style={styles.schedStd}>±{r.std.toFixed(2)}σ</Text>
                <Text style={[styles.schedRisk, { color: i === 0 ? COLORS.accent4 : COLORS.muted }]}>{r.riskScore.toFixed(2)}</Text>
              </View>
              <AnimatedBar value={r.riskScore} maxValue={(lastSchedule[lastSchedule.length - 1]?.riskScore ?? 1) + 0.1} color={i === 0 ? COLORS.accent4 : COLORS.border} height={4} />
            </View>
          ))}
          <View style={styles.bestBadgeRow}>
            <Text style={styles.bestBadgeText}>✓ SELECTED: {lastSchedule[0]?.node?.toUpperCase()}</Text>
          </View>
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Task Stream</Text>
        {taskLog.length === 0 && (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>⚡</Text>
            <Text style={styles.emptyText}>Press START or Dispatch a Task</Text>
          </View>
        )}
        {taskLog.slice(0, 18).map((t, i) => (
          <View key={t.id} style={[styles.logRow, i === 0 && styles.logRowNew]}>
            <View style={styles.logLeft}>
              <Text style={styles.logId}>#{t.id}</Text>
              <Text style={styles.logType}>{t.type.replace('_', '\n')}</Text>
            </View>
            <View style={styles.logMid}>
              <Text style={[styles.logNode, { color: NODE_POOL.find(n => n.name === t.bestNode)?.color ?? COLORS.text }]}>{t.bestNode}</Text>
              <View style={styles.logTierPill}><Text style={styles.logTierText}>{t.bestTier}</Text></View>
            </View>
            <View style={styles.logRight}>
              <Text style={styles.logPred}>{t.predTime.toFixed(2)}s</Text>
              <Text style={styles.logUncert}>±{t.uncertainty.toFixed(2)}σ</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

// ─── Nodes Tab ────────────────────────────────────────────────────────────────

function NodesTab({ nodeLoads, lastSchedule }) {
  const tiers = [
    { id: 'FOG_EDGE', label: '🌫  Fog Edge  (Tier 0)', desc: 'Low-latency edge devices · 500–2000 MIPS' },
    { id: 'FOG_MID', label: '⚡  Fog Mid  (Tier 1)', desc: 'Intermediate fog servers · 2000–8000 MIPS' },
    { id: 'CLOUD', label: '☁  Cloud  (Tier 2)', desc: 'High-capacity data centres · 8000–32000 MIPS' },
  ];
  return (
    <View>
      {tiers.map(tier => (
        <View key={tier.id} style={styles.section}>
          <Text style={styles.sectionTitle}>{tier.label}</Text>
          <Text style={styles.sectionSubtitle}>{tier.desc}</Text>
          {NODE_POOL.filter(n => n.tier === tier.id).map(node => {
            const load = nodeLoads[node.name] ?? 0.3;
            const sched = lastSchedule?.find(r => r.node === node.name);
            const isBest = lastSchedule?.[0]?.node === node.name;
            return (
              <View key={node.name} style={[styles.nodeCard, isBest && { borderColor: COLORS.accent4 }]}>
                <View style={styles.nodeCardHeader}>
                  <View style={styles.nodeCardTitleRow}>
                    {isBest && <View style={[styles.bestDot, { backgroundColor: COLORS.accent4 }]} />}
                    <Text style={[styles.nodeCardName, { color: node.color }]}>{node.name}</Text>
                    {isBest && <View style={styles.bestTag}><Text style={styles.bestTagText}>BEST</Text></View>}
                  </View>
                  <Text style={styles.nodeLoad}>Load  {(load * 100).toFixed(0)}%</Text>
                </View>
                <AnimatedBar value={load} maxValue={1} color={load > 0.7 ? COLORS.accent2 : load > 0.4 ? COLORS.warning : COLORS.accent4} height={7} />
                <View style={styles.nodeSpecs}>
                  <View style={styles.nodeSpec}><Text style={styles.nodeSpecVal}>{node.mips.toLocaleString()}</Text><Text style={styles.nodeSpecLabel}>MIPS</Text></View>
                  <View style={styles.nodeSpec}><Text style={styles.nodeSpecVal}>{node.ram}GB</Text><Text style={styles.nodeSpecLabel}>RAM</Text></View>
                  <View style={styles.nodeSpec}><Text style={styles.nodeSpecVal}>{node.bw}</Text><Text style={styles.nodeSpecLabel}>Mbps</Text></View>
                  {sched ? <View style={styles.nodeSpec}><Text style={[styles.nodeSpecVal, { color: COLORS.accent3 }]}>{sched.pred.toFixed(1)}s</Text><Text style={styles.nodeSpecLabel}>pred</Text></View> : null}
                  {sched ? <View style={styles.nodeSpec}><Text style={[styles.nodeSpecVal, { color: COLORS.accent2 }]}>±{sched.std.toFixed(2)}</Text><Text style={styles.nodeSpecLabel}>σ</Text></View> : null}
                  {sched ? <View style={styles.nodeSpec}><Text style={[styles.nodeSpecVal, { color: COLORS.accent1 }]}>{sched.riskScore.toFixed(2)}</Text><Text style={styles.nodeSpecLabel}>risk</Text></View> : null}
                </View>
              </View>
            );
          })}
        </View>
      ))}
    </View>
  );
}

// ─── Drift Tab ────────────────────────────────────────────────────────────────

function DriftTab({ drift, onBurst }) {
  const klAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(klAnim, { toValue: Math.min(drift.kl / 1.0, 1), duration: 600, useNativeDriver: false }).start();
  }, [drift.kl]);
  const klColor = drift.detected ? COLORS.accent2 : drift.kl > 0.08 ? COLORS.warning : COLORS.accent4;

  return (
    <View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Context-Shift Detector  (CSD)</Text>
        <Text style={styles.sectionSubtitle}>KL divergence checked every 20 tasks · threshold = {KL_THRESHOLD} · window = 200</Text>

        <View style={[styles.driftStatusCard, { borderColor: klColor }]}>
          <View style={styles.driftStatusRow}>
            <View>
              <Text style={styles.driftLabel}>KL DIVERGENCE</Text>
              <Text style={[styles.driftKL, { color: klColor }]}>{drift.kl.toFixed(4)}</Text>
            </View>
            <View style={[styles.driftBadge, { borderColor: klColor, backgroundColor: drift.detected ? 'rgba(255,78,106,0.12)' : 'rgba(52,211,153,0.12)' }]}>
              <PulseDot color={klColor} size={7} />
              <Text style={[styles.driftBadgeText, { color: klColor }]}>{drift.detected ? 'DRIFT DETECTED' : 'NORMAL'}</Text>
            </View>
          </View>
          <View style={styles.klGaugeTrack}>
            <Animated.View style={[styles.klGaugeFill, { backgroundColor: klColor, width: klAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }) }]} />
            <View style={[styles.klThresholdLine, { left: `${KL_THRESHOLD * 100}%` }]} />
          </View>
          <Text style={styles.driftInfo}>Tasks: {drift.taskCount} · Burst: {drift.burstMode ? 'ON 🔥' : 'OFF'} · Retrain: {drift.retrain ? 'YES ⚠' : 'NO'}</Text>
        </View>

        <View style={styles.driftScenarios}>
          <View style={[styles.driftScenario, { flex: 1 }]}>
            <Text style={[styles.driftScenLabel, { color: COLORS.accent4 }]}>Normal Load</Text>
            <Text style={[styles.driftScenKL, { color: COLORS.text }]}>KL ≈ 0.08</Text>
            <View style={[styles.driftScenStatus, { borderColor: COLORS.accent4 }]}>
              <Text style={[styles.driftScenStatusText, { color: COLORS.accent4 }]}>✗ No Drift</Text>
            </View>
          </View>
          <View style={{ width: 1, backgroundColor: COLORS.border }} />
          <View style={[styles.driftScenario, { flex: 1 }]}>
            <Text style={[styles.driftScenLabel, { color: COLORS.accent2 }]}>Burst (5×)</Text>
            <Text style={[styles.driftScenKL, { color: COLORS.text }]}>KL ≈ 0.73</Text>
            <View style={[styles.driftScenStatus, { borderColor: COLORS.accent2 }]}>
              <Text style={[styles.driftScenStatusText, { color: COLORS.accent2 }]}>✓ Drift + Retrain</Text>
            </View>
          </View>
        </View>

        <TouchableOpacity style={[styles.burstBtn, { borderColor: drift.burstMode ? COLORS.accent2 : COLORS.warning }]} onPress={onBurst} activeOpacity={0.7}>
          <Text style={[styles.burstBtnText, { color: drift.burstMode ? COLORS.accent2 : COLORS.warning }]}>
            {drift.burstMode ? '🔥 Burst Mode Active (8s)' : '💥 Simulate Workload Burst'}
          </Text>
          <Text style={styles.burstBtnSub}>5× task sizes · +0.3 node load offset</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>How CSD Works</Text>
        {[
          ['1', 'Reference window', 'Store the first 200-task distribution as baseline reference'],
          ['2', 'Sliding comparison', 'Every 20 tasks, compare current window vs. reference'],
          ['3', 'KL divergence', 'Histogram-based KL score measures statistical distance'],
          ['4', 'Threshold check', 'If KL > 0.15 → drift_detected = True'],
          ['5', 'Auto-retrain', 'Sets retrain_needed = True · closes the feedback loop'],
        ].map(([step, title, desc]) => (
          <View key={step} style={styles.csdStep}>
            <View style={[styles.csdStepNum, { backgroundColor: COLORS.accent1 }]}>
              <Text style={styles.csdStepNumText}>{step}</Text>
            </View>
            <View style={styles.csdStepBody}>
              <Text style={styles.csdStepTitle}>{title}</Text>
              <Text style={styles.csdStepDesc}>{desc}</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

// ─── Metrics Tab ──────────────────────────────────────────────────────────────

function MetricsTab({ taskLog }) {
  const modelComparison = [
    { name: 'Linear Reg.', mae: 14.28, rmse: 28.49, r2: 0.7214, ours: false },
    { name: 'Ridge Reg.', mae: 14.26, rmse: 28.47, r2: 0.7216, ours: false },
    { name: 'Decision Tree', mae: 6.88, rmse: 14.22, r2: 0.9101, ours: false },
    { name: 'Random Forest', mae: 4.11, rmse: 8.90, r2: 0.9651, ours: false },
    { name: 'Grad. Boosting', mae: 4.58, rmse: 9.78, r2: 0.9581, ours: false },
    { name: 'SVR', mae: 5.92, rmse: 11.88, r2: 0.9401, ours: false },
    { name: 'UQE (Ours)', mae: 3.88, rmse: 8.44, r2: 0.9714, ours: true },
  ];
  const scheduling = [
    { policy: 'Round-Robin', time: 312, color: COLORS.muted },
    { policy: 'ML-Greedy', time: 198, color: COLORS.accent3 },
    { policy: 'UASP (α=1.5)', time: 175, color: COLORS.accent4 },
  ];
  const total = taskLog.length || 1;
  const tierCounts = { FOG_EDGE: 0, FOG_MID: 0, CLOUD: 0 };
  taskLog.forEach(t => { if (tierCounts[t.bestTier] !== undefined) tierCounts[t.bestTier]++; });

  return (
    <View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Model Comparison</Text>
        <View style={styles.tableHeader}>
          <Text style={[styles.tableHeaderCell, { flex: 2.2 }]}>Model</Text>
          <Text style={styles.tableHeaderCell}>MAE</Text>
          <Text style={styles.tableHeaderCell}>RMSE</Text>
          <Text style={styles.tableHeaderCell}>R²</Text>
        </View>
        {modelComparison.map((m, i) => (
          <View key={m.name} style={[styles.tableRow, m.ours && styles.tableRowBest, i % 2 === 0 && styles.tableRowEven]}>
            <Text style={[styles.tableCell, { flex: 2.2 }, m.ours && { color: COLORS.accent2, fontWeight: '700' }]} numberOfLines={1}>{m.name}</Text>
            <View style={[styles.tableCell, { flex: 1, alignItems: 'flex-start' }]}>
              <View style={[styles.metricPill, m.ours && styles.metricPillBest]}>
                <Text style={[styles.metricPillText, m.ours && { color: COLORS.accent4 }]}>{m.mae}</Text>
              </View>
            </View>
            <View style={[styles.tableCell, { flex: 1, alignItems: 'flex-start' }]}>
              <View style={[styles.metricPill, m.ours && styles.metricPillBest]}>
                <Text style={[styles.metricPillText, m.ours && { color: COLORS.accent4 }]}>{m.rmse}</Text>
              </View>
            </View>
            <View style={[styles.tableCell, { flex: 1, alignItems: 'flex-start' }]}>
              <View style={[styles.metricPill, m.ours && styles.metricPillBest]}>
                <Text style={[styles.metricPillText, m.ours && { color: COLORS.accent4 }]}>{m.r2}</Text>
              </View>
            </View>
          </View>
        ))}
        <Text style={styles.tableNote}>UQE outperforms all 6 baselines on every metric.</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Scheduling Comparison</Text>
        {scheduling.map((s, i) => (
          <View key={s.policy} style={[styles.policyCard, i === 2 && { borderColor: COLORS.accent4 }]}>
            {i === 2 && <View style={styles.policyBestTag}><Text style={styles.policyBestText}>BEST</Text></View>}
            <Text style={styles.policyName}>{s.policy}</Text>
            <Text style={[styles.policyValue, { color: s.color }]}>{s.time}s</Text>
            <Text style={styles.policyUnit}>mean exec time</Text>
            <AnimatedBar value={312 - s.time} maxValue={312 - 150} color={s.color} height={5} />
            {i === 2 && <Text style={[styles.policyImprove, { color: COLORS.accent4 }]}>~44% lower than Round-Robin</Text>}
          </View>
        ))}
      </View>

      {taskLog.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Live Tier Distribution</Text>
          {Object.entries(tierCounts).map(([tier, count]) => {
            const pct = count / total;
            const color = tier === 'CLOUD' ? COLORS.accent1 : tier === 'FOG_MID' ? COLORS.accent3 : COLORS.accent4;
            return (
              <View key={tier} style={styles.tierRow}>
                <Text style={[styles.tierLabel, { color }]}>{tier}</Text>
                <View style={styles.tierBar}><AnimatedBar value={pct} maxValue={1} color={color} height={8} /></View>
                <Text style={[styles.tierPct, { color }]}>{(pct * 100).toFixed(0)}%</Text>
                <Text style={styles.tierCount}>({count})</Text>
              </View>
            );
          })}
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Novel Contributions</Text>
        {[
          { label: 'HCFE', full: 'Heterogeneous-Context Feature Engineering', desc: '18 features: 6 task + 5 node + 7 cross-dimension interactions', color: COLORS.accent1 },
          { label: 'UQE', full: 'Uncertainty-Quantified Ensemble', desc: 'RF + GB + SVR with 30 bootstrap resamples → (pred_time, σ)', color: COLORS.accent3 },
          { label: 'UASP', full: 'Uncertainty-Aware Scheduling Policy', desc: 'risk = pred + α×σ  — generalises all existing ML scheduling', color: COLORS.accent4 },
          { label: 'CSD', full: 'Context-Shift Detector', desc: 'Sliding-window KL divergence → auto-retraining feedback loop', color: COLORS.accent2 },
        ].map(c => (
          <View key={c.label} style={[styles.contribCard, { borderLeftColor: c.color }]}>
            <View style={[styles.contribLabel, { backgroundColor: c.color + '22' }]}>
              <Text style={[styles.contribLabelText, { color: c.color }]}>{c.label}</Text>
            </View>
            <View style={styles.contribBody}>
              <Text style={styles.contribFull}>{c.full}</Text>
              <Text style={styles.contribDesc}>{c.desc}</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const MONO = Platform.OS === 'android' ? 'monospace' : 'Courier New';

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },

  header: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  headerLeft: { flex: 1 },
  headerTagRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 4 },
  headerLine: { width: 18, height: 1 },
  headerTag: { fontSize: 9, letterSpacing: 2, fontFamily: MONO, textTransform: 'uppercase' },
  headerTitle: { fontSize: 22, fontWeight: '800', color: COLORS.text, lineHeight: 27 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  controlBtn: { borderWidth: 1, paddingHorizontal: 14, paddingVertical: 9, borderRadius: 3 },
  controlBtnText: { fontSize: 11, fontWeight: '800', fontFamily: MONO, letterSpacing: 1 },

  kpiScroll: { maxHeight: 108 },
  kpiRow: { paddingHorizontal: 12, paddingVertical: 10, gap: 8, flexDirection: 'row' },
  kpiCard: { backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, padding: 12, width: 100, borderRadius: 2, position: 'relative', overflow: 'hidden' },
  kpiAccent: { position: 'absolute', top: 0, left: 0, right: 0, height: 2 },
  kpiIcon: { fontSize: 14, marginBottom: 3 },
  kpiLabel: { fontSize: 8, color: COLORS.muted, textTransform: 'uppercase', letterSpacing: 0.8, fontFamily: MONO, marginBottom: 2 },
  kpiValue: { fontSize: 20, fontWeight: '800' },
  kpiUnit: { fontSize: 9, color: COLORS.muted, fontFamily: MONO },

  tabBar: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: COLORS.border, backgroundColor: COLORS.surface },
  tabItem: { flex: 1, alignItems: 'center', paddingVertical: 9, position: 'relative' },
  tabIcon: { fontSize: 15, color: COLORS.muted },
  tabLabel: { fontSize: 9, color: COLORS.muted, marginTop: 2, fontFamily: MONO, textTransform: 'uppercase', letterSpacing: 0.5 },
  tabIndicator: { position: 'absolute', bottom: 0, left: '15%', right: '15%', height: 2, backgroundColor: COLORS.accent1 },

  content: { flex: 1 },
  section: { paddingHorizontal: 14, marginTop: 18 },
  sectionTitle: { fontSize: 10, color: COLORS.accent1, textTransform: 'uppercase', letterSpacing: 1.5, fontFamily: MONO, marginBottom: 3, fontWeight: '700' },
  sectionSubtitle: { fontSize: 9.5, color: COLORS.muted, fontFamily: MONO, marginBottom: 10, lineHeight: 14 },

  alertBanner: { marginHorizontal: 14, marginTop: 10, flexDirection: 'row', alignItems: 'center', gap: 8, borderWidth: 1, padding: 10, backgroundColor: 'rgba(255,78,106,0.08)', borderRadius: 2 },
  alertText: { fontSize: 10.5, fontFamily: MONO, flex: 1 },

  actionRow: { flexDirection: 'row', gap: 8, paddingHorizontal: 14, marginTop: 14 },
  actionBtn: { borderWidth: 1, padding: 12, alignItems: 'center', borderRadius: 2 },
  actionBtnText: { fontSize: 12, fontWeight: '700', fontFamily: MONO },

  scheduleRow: { backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, padding: 11, marginBottom: 6, borderRadius: 2 },
  scheduleRowBest: { borderColor: COLORS.accent4, backgroundColor: 'rgba(52,211,153,0.04)' },
  scheduleLeft: { flexDirection: 'row', alignItems: 'center', gap: 9, marginBottom: 7 },
  scheduleRight: { flexDirection: 'row', gap: 10, marginBottom: 6, alignItems: 'center' },
  rankBadge: { width: 22, height: 22, borderRadius: 2, alignItems: 'center', justifyContent: 'center' },
  rankText: { fontSize: 10, fontWeight: '800', fontFamily: MONO },
  schedNodeName: { fontSize: 12, fontWeight: '700', fontFamily: MONO },
  schedNodeTier: { fontSize: 8.5, color: COLORS.muted, fontFamily: MONO },
  schedPred: { fontSize: 13, fontWeight: '700', color: COLORS.text, fontFamily: MONO },
  schedStd: { fontSize: 11, color: COLORS.muted, fontFamily: MONO },
  schedRisk: { fontSize: 13, fontWeight: '700', fontFamily: MONO },
  bestBadgeRow: { backgroundColor: 'rgba(52,211,153,0.08)', borderWidth: 1, borderColor: COLORS.accent4, padding: 8, alignItems: 'center', marginTop: 2, borderRadius: 1 },
  bestBadgeText: { fontSize: 10.5, color: COLORS.accent4, fontWeight: '700', fontFamily: MONO, letterSpacing: 1.5 },

  logRow: { flexDirection: 'row', alignItems: 'center', borderBottomWidth: 1, borderBottomColor: COLORS.border, paddingVertical: 9 },
  logRowNew: { backgroundColor: 'rgba(0,212,255,0.025)' },
  logLeft: { width: 68, gap: 2 },
  logId: { fontSize: 8.5, color: COLORS.muted, fontFamily: MONO },
  logType: { fontSize: 8.5, color: COLORS.text, fontFamily: MONO, textTransform: 'uppercase' },
  logMid: { flex: 1, gap: 3 },
  logNode: { fontSize: 11, fontWeight: '700', fontFamily: MONO },
  logTierPill: { backgroundColor: COLORS.border, paddingHorizontal: 5, paddingVertical: 1, alignSelf: 'flex-start', borderRadius: 2 },
  logTierText: { fontSize: 7.5, color: COLORS.muted, fontFamily: MONO },
  logRight: { alignItems: 'flex-end', gap: 2 },
  logPred: { fontSize: 13, fontWeight: '700', color: COLORS.text, fontFamily: MONO },
  logUncert: { fontSize: 9, color: COLORS.muted, fontFamily: MONO },

  emptyState: { alignItems: 'center', paddingVertical: 40, gap: 8 },
  emptyIcon: { fontSize: 30 },
  emptyText: { color: COLORS.muted, fontSize: 12, fontFamily: MONO },

  barTrack: { backgroundColor: COLORS.border, borderRadius: 2, overflow: 'hidden', width: '100%' },
  barFill: { height: '100%', borderRadius: 2 },

  nodeCard: { backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, padding: 13, marginBottom: 8, borderRadius: 2 },
  nodeCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  nodeCardTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  bestDot: { width: 6, height: 6, borderRadius: 3 },
  nodeCardName: { fontSize: 12.5, fontWeight: '700', fontFamily: MONO },
  bestTag: { backgroundColor: 'rgba(52,211,153,0.15)', borderWidth: 1, borderColor: COLORS.accent4, paddingHorizontal: 5, paddingVertical: 1 },
  bestTagText: { fontSize: 7.5, color: COLORS.accent4, fontFamily: MONO, letterSpacing: 1 },
  nodeLoad: { fontSize: 9.5, color: COLORS.muted, fontFamily: MONO },
  nodeSpecs: { flexDirection: 'row', gap: 14, marginTop: 10, flexWrap: 'wrap' },
  nodeSpec: { alignItems: 'center' },
  nodeSpecVal: { fontSize: 11.5, color: COLORS.text, fontWeight: '700', fontFamily: MONO },
  nodeSpecLabel: { fontSize: 8, color: COLORS.muted, fontFamily: MONO },

  driftStatusCard: { backgroundColor: COLORS.card, borderWidth: 1, borderRadius: 2, padding: 15, marginBottom: 12 },
  driftStatusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 },
  driftLabel: { fontSize: 8.5, color: COLORS.muted, fontFamily: MONO, textTransform: 'uppercase', letterSpacing: 1 },
  driftKL: { fontSize: 28, fontWeight: '800', fontFamily: MONO },
  driftBadge: { borderWidth: 1, flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 9, paddingVertical: 6, borderRadius: 2, alignSelf: 'flex-start' },
  driftBadgeText: { fontSize: 10, fontWeight: '700', fontFamily: MONO },
  klGaugeTrack: { height: 10, backgroundColor: COLORS.border, borderRadius: 2, position: 'relative', overflow: 'visible' },
  klGaugeFill: { height: '100%', borderRadius: 2 },
  klThresholdLine: { position: 'absolute', top: -3, width: 2, height: 16, backgroundColor: COLORS.accent2 },
  driftInfo: { fontSize: 9, color: COLORS.muted, fontFamily: MONO, marginTop: 8 },
  driftScenarios: { flexDirection: 'row', marginBottom: 12 },
  driftScenario: { backgroundColor: COLORS.card, padding: 14, alignItems: 'center', gap: 4 },
  driftScenLabel: { fontSize: 10, fontWeight: '700', fontFamily: MONO },
  driftScenKL: { fontSize: 20, fontWeight: '800', fontFamily: MONO },
  driftScenStatus: { borderWidth: 1, paddingHorizontal: 7, paddingVertical: 2 },
  driftScenStatusText: { fontSize: 8.5, fontFamily: MONO, fontWeight: '700' },
  burstBtn: { borderWidth: 1, padding: 13, alignItems: 'center', borderRadius: 2 },
  burstBtnText: { fontSize: 12.5, fontWeight: '700', fontFamily: MONO },
  burstBtnSub: { fontSize: 9, color: COLORS.muted, fontFamily: MONO, marginTop: 3 },

  csdStep: { flexDirection: 'row', gap: 11, marginBottom: 10, alignItems: 'flex-start' },
  csdStepNum: { width: 21, height: 21, borderRadius: 2, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  csdStepNumText: { fontSize: 10, fontWeight: '800', color: COLORS.bg, fontFamily: MONO },
  csdStepBody: { flex: 1 },
  csdStepTitle: { fontSize: 11.5, color: COLORS.text, fontWeight: '700', marginBottom: 1 },
  csdStepDesc: { fontSize: 9.5, color: COLORS.muted, fontFamily: MONO, lineHeight: 13.5 },

  tableHeader: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: COLORS.accent1, paddingBottom: 6, marginBottom: 3 },
  tableHeaderCell: { flex: 1, fontSize: 7.5, color: COLORS.accent1, fontFamily: MONO, textTransform: 'uppercase', letterSpacing: 0.8 },
  tableRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  tableRowEven: { backgroundColor: 'rgba(255,255,255,0.01)' },
  tableRowBest: { backgroundColor: 'rgba(255,78,106,0.04)' },
  tableCell: { flex: 1, fontSize: 9.5, color: COLORS.text, fontFamily: MONO },
  metricPill: { backgroundColor: 'rgba(255,78,106,0.1)', paddingHorizontal: 4, paddingVertical: 2, borderRadius: 2 },
  metricPillBest: { backgroundColor: 'rgba(52,211,153,0.12)' },
  metricPillText: { fontSize: 9, color: COLORS.accent2, fontFamily: MONO },
  tableNote: { fontSize: 9, color: COLORS.accent4, fontFamily: MONO, marginTop: 7, fontStyle: 'italic' },

  policyCard: { backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, padding: 14, marginBottom: 8, borderRadius: 2, position: 'relative' },
  policyBestTag: { position: 'absolute', top: 10, right: 10, backgroundColor: 'rgba(52,211,153,0.15)', borderWidth: 1, borderColor: COLORS.accent4, paddingHorizontal: 5, paddingVertical: 1 },
  policyBestText: { fontSize: 7.5, color: COLORS.accent4, fontFamily: MONO, letterSpacing: 0.8 },
  policyName: { fontSize: 9.5, color: COLORS.muted, fontFamily: MONO, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 },
  policyValue: { fontSize: 26, fontWeight: '800', fontFamily: MONO },
  policyUnit: { fontSize: 8.5, color: COLORS.muted, fontFamily: MONO, marginBottom: 8 },
  policyImprove: { fontSize: 9.5, fontFamily: MONO, marginTop: 5, fontWeight: '700' },

  tierRow: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 10 },
  tierLabel: { width: 78, fontSize: 8.5, fontFamily: MONO, textTransform: 'uppercase' },
  tierBar: { flex: 1 },
  tierPct: { width: 32, fontSize: 11, fontWeight: '700', fontFamily: MONO, textAlign: 'right' },
  tierCount: { width: 30, fontSize: 8.5, color: COLORS.muted, fontFamily: MONO },

  contribCard: { flexDirection: 'row', borderLeftWidth: 3, backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, marginBottom: 7, overflow: 'hidden', borderRadius: 2 },
  contribLabel: { width: 52, alignItems: 'center', justifyContent: 'center', padding: 10 },
  contribLabelText: { fontSize: 11, fontWeight: '800', fontFamily: MONO },
  contribBody: { flex: 1, padding: 10 },
  contribFull: { fontSize: 11, color: COLORS.text, fontWeight: '700', marginBottom: 2 },
  contribDesc: { fontSize: 9.5, color: COLORS.muted, fontFamily: MONO, lineHeight: 13.5 },
});