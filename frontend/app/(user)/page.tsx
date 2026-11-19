'use client'

import { useEffect, useId, useState } from "react"
import { Activity, History, CheckCircle2, XCircle, AlertTriangle, AlertCircle, Info, Cpu, HardDrive, Zap, Layers, Server, Sparkles, ShieldCheck } from "lucide-react"
import CustomSelect from "@/app/components/CustomSelect"
import Drawer from "@/app/components/Drawer"
import Loading from "@/app/components/Loading"
import ProviderLogo from "@/app/components/ProviderLogo"
import StatusBadge from "@/app/components/StatusBadge"
import {
    automaticRam,
    CALCULATION_VERSION,
    calculateCapacityFloor,
    calculateStandardGpu,
    GB,
    resolveKvCacheBits,
    resolveWeightPrecision,
    WEIGHT_BITS,
    type KvPrecisionChoice,
    type ModelMetadata,
    type WeightPrecisionChoice,
} from "@/app/lib/calculators"

type Model = ModelMetadata & { id: number, model: string, provider: string, parametersB: number, modelType?: string | null }
type Gpu = { id: string, vendor: string, name: string, vramGb: number }
type Catalog = { models: Model[], gpus: Gpu[] }
type Result = ReturnType<typeof calculateStandardGpu>
type Evaluation = { result: Result | null, error: string }
const tokenPresets = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
const statusRank = { fits: 0, unknown: 1, does_not_fit: 2 }

async function request(url: string, options?: RequestInit) {
    const response = await fetch(url, options)
    const data = await response.json()
    if (!response.ok) throw new Error(data.message ?? `Request failed (${response.status})`)
    return data
}

function capacityEstimate(workloadBytes: number, gpu: Gpu, targetUtilization: string) {
    try {
        return calculateCapacityFloor(
            workloadBytes,
            gpu.vramGb,
            Number(targetUtilization),
        )
    } catch {
        return null
    }
}

export default function LlmSizerPage() {
    const [catalog, setCatalog] = useState<Catalog | null>(null)
    const [error, setError] = useState("")
    const [modelId, setModelId] = useState("")
    const [gpuId, setGpuId] = useState("")
    const [hardwareMode, setHardwareMode] = useState<"gpu" | "dgx_spark">("gpu")
    const [gpuCount, setGpuCount] = useState("1")
    const [ram, setRam] = useState("0")
    const [ramDirty, setRamDirty] = useState(false)
    const [ramMultiplier, setRamMultiplier] = useState("2")
    const [weightPrecision, setWeightPrecision] = useState<WeightPrecisionChoice>("model")
    const [kvPrecision, setKvPrecision] = useState<KvPrecisionChoice>("model")
    const [tokenPreset, setTokenPreset] = useState("4096")
    const [inputTokens, setInputTokens] = useState("4096")
    const [concurrent, setConcurrent] = useState("1")
    const [growthPercent, setGrowthPercent] = useState("20")
    const [runtimeGb, setRuntimeGb] = useState("4")
    const [targetUtilization, setTargetUtilization] = useState("85")
    const [historyOpen, setHistoryOpen] = useState(false)
    const [history, setHistory] = useState<Record<string, unknown>[]>([])
    const [saving, setSaving] = useState(false)

    useEffect(() => {
        request("/api/catalog").then((data: Catalog) => {
            setCatalog(data)
            const initialModel = data.models.find((item) => isMemoryReady(item) && isLlmModel(item))
                ?? data.models.find(isLlmModel)
                ?? data.models[0]
            setModelId(String(initialModel?.id ?? ""))
            setGpuId(data.gpus[0]?.id ?? "")
            setRam(String((data.gpus[0]?.vramGb ?? 0) * 2))
        }).catch((e) => setError(e.message))
    }, [])

    const gpu = catalog?.gpus.find((item) => item.id === gpuId)
    const isDgxSpark = hardwareMode === "dgx_spark"
    const model = catalog?.models.find((item) => String(item.id) === modelId)
    const weightDefaultLabel = model?.weightFormat && model.weightBits
        ? `Model Default (${model.weightFormat}, ${model.weightBits}-bit)`
        : "Model Default (FP16 fallback)"
    const kvDefaultLabel = model?.kvCacheBits
        ? `Model Default (${model.kvCacheBits}-bit)`
        : "Model Default (16-bit fallback)"
    const effectiveConcurrent = Math.ceil(Number(concurrent) * (1 + Number(growthPercent) / 100))

    function evaluate(item: Model, selectedGpu = gpu, count = Number(gpuCount)): Evaluation {
        try {
            if (!selectedGpu) throw new Error("Select a GPU")
            if (ram === "" || !Number.isFinite(Number(ram)) || Number(ram) < 0) throw new Error("Host RAM must be a finite nonnegative number")
            const resolvedWeight = resolveWeightPrecision(item, weightPrecision)
            const resolvedKv = resolveKvCacheBits(item, kvPrecision)
            const numericFields = [inputTokens, concurrent, growthPercent, runtimeGb, ...(isDgxSpark ? [] : [targetUtilization]), gpuCount]
            if (numericFields.some((value) => !value.trim())) throw new Error("Complete the numeric workload and policy inputs")
            if (isDgxSpark && ![1, 2].includes(count)) throw new Error("DGX Spark count must be 1 or 2")
            if (Number(growthPercent) < 0 || Number(growthPercent) > 100) throw new Error("Growth capacity must be between 0 and 100 percent")
            if (Number(targetUtilization) <= 0 || Number(targetUtilization) > 100) throw new Error("Target utilization must be above 0 and at most 100 percent")
            const input = {
                model: item,
                weightFormat: resolvedWeight.format,
                weightBits: resolvedWeight.bits,
                kvCacheBits: resolvedKv.bits,
                inputTokens: Number(inputTokens),
                maxOutputTokens: 0,
                concurrentSequences: effectiveConcurrent,
                servingEngine: "",
                engineVersion: "",
                runtimePeakBytes: Number(runtimeGb) * GB,
                quantizationOverheadPercent: 15,
                partition: null,
                targetTTFTMs: null,
                targetTPOTMs: null,
                targetOutputTokensPerSecondPerRequest: null,
                hardware: {
                    memoryGb: selectedGpu.vramGb,
                    deviceCount: count,
                    systemReservedGb: 0, safetyReservePercent: isDgxSpark ? 0 : 100 - Number(targetUtilization),
                },
            }
            return { result: calculateStandardGpu(input), error: "" }
        } catch (e) {
            return { result: null, error: e instanceof Error ? e.message : "Invalid sizing input" }
        }
    }

    const live = model ? evaluate(model) : null
    // Candidate search intentionally has no invented partition. Multi-device candidates remain Unknown.
    const candidates = model && !isDgxSpark ? (catalog?.gpus.filter((item) => item.id !== "dgx_spark").flatMap((item) => [1, 2, 4, 8].map((count) => ({ gpu: item, count, ...evaluate(model, item, count) }))) ?? [])
        .sort((a, b) => (statusRank[a.result?.memoryStatus ?? "unknown"] - statusRank[b.result?.memoryStatus ?? "unknown"]) || a.gpu.vramGb * a.count - b.gpu.vramGb * b.count).slice(0, 3) : []

    function changeHardwareMode(mode: "gpu" | "dgx_spark") {
        setHardwareMode(mode)
        const selected = mode === "dgx_spark"
            ? catalog?.gpus.find((item) => item.id === "dgx_spark")
            : catalog?.gpus.find((item) => item.id !== "dgx_spark")
        setGpuId(selected?.id ?? "")
        setGpuCount("1")
    }

    function changeHardware(id: string, count: string) {
        setGpuId(id)
        setGpuCount(count)
        const selected = catalog?.gpus.find((item) => item.id === id)
        if (selected) setRam(automaticRam(ram, ramDirty, selected.vramGb, Number(count), Number(ramMultiplier)))
    }

    function changeTokenPreset(value: string) {
        setTokenPreset(value)
        if (value !== "custom") setInputTokens(value)
    }

    async function loadHistory() {
        try { setHistory(await request("/api/history/llm-pick-model")) } catch (e) { setError(e instanceof Error ? e.message : "History failed") }
    }

    async function save() {
        if (saving || !model || !live?.result) return
        const name = window.prompt("History name")?.trim()
        if (!name) return
        setSaving(true)
        setError("")
        try {
            const result = live.result
            const hardwareType = isDgxSpark ? "dgx_spark" : "standard"
            const systemRamGb = isDgxSpark ? null : Number(ram)
            const body = {
                name,
                hardwareType,
                useCase: "llm",
                contextTokens: Number(inputTokens),
                quant: result.inputSnapshot.weightFormat,
                concurrent: effectiveConcurrent,
                gpuId: gpu?.id,
                gpuName: gpu?.name,
                gpuCount: Number(gpuCount),
                systemRamGb,
                calculationVersion: CALCULATION_VERSION,
                calculationSnapshot: JSON.stringify({
                    version: CALCULATION_VERSION,
                    hardwareType,
                    hardwareId: gpu?.id,
                    hardwareName: gpu?.name,
                    ram: systemRamGb,
                    ramMultiplier,
                    growthPercent: Number(growthPercent),
                    targetUtilization: isDgxSpark ? null : Number(targetUtilization),
                    calculations: [{ modelName: model.model, input: result.inputSnapshot, result }],
                }),
                modelId: model.id,
                modelName: model.model,
                modelProvider: model.provider,
                modelParametersB: model.parametersB,
                status: result.memoryStatus,
                recommendationText: "Memory estimate only; compatibility unknown; performance not benchmarked.",
                weightGb: result.breakdown.weights === null ? null : result.breakdown.weights / GB,
                kvGb: result.breakdown.kvCache === null ? null : result.breakdown.kvCache / GB,
            }
            await request("/api/history/llm-pick-model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
        } catch (e) { setError(e instanceof Error ? e.message : "Save failed") } finally { setSaving(false) }
    }

    if (!catalog) return error ? <p role="alert">{error}</p> : <Loading />
    return (
        <section className="page-stack">
            <div className="page-title-row">
                <div><h1>LLM Hardware Sizer</h1><p className="muted">Inference memory estimates, not validated configurations or performance guarantees.</p></div>
                <div className="button-row">
                    <button type="button" className="button secondary" onClick={() => { setHistoryOpen(true); loadHistory() }}><History size={16} /> History</button>
                </div>
            </div>
            {error ? <p role="alert">{error}</p> : null}
            <div className="workbench-grid">
                <div className="page-stack">
                    <div className="panel page-stack sizing-parameters-card">
                        <section className="sizer-step" aria-labelledby="model-step-title">
                            <StepHeading number="1" title="Model" detail="Identity and precision" id="model-step-title" />
                            <div className="field"><label>Target Model</label><CustomSelect ariaLabel="Target Model" value={modelId} searchable options={catalog.models.filter(isLlmModel).map((item) => ({ value: String(item.id), label: `${item.model}${metadataStatus(item) === "ready" ? "" : ` - ${metadataStatus(item)}`}`, icon: <ProviderLogo name={item.provider} /> }))} onChange={setModelId} /></div>
                            <div className="grid-2">
                                <div className="field">
                                    <label>Weight Precision</label>
                                    <CustomSelect
                                        ariaLabel="Weight Precision"
                                        value={weightPrecision}
                                        options={[
                                            { value: "model", label: weightDefaultLabel },
                                            ...Object.keys(WEIGHT_BITS).map((value) => ({ value, label: value === "FP16" ? "FP16 (Half precision)" : value })),
                                        ]}
                                        onChange={(value) => setWeightPrecision(value as WeightPrecisionChoice)}
                                    />
                                </div>
                                <div className="field">
                                    <label>Attention Cache Precision</label>
                                    <CustomSelect
                                        ariaLabel="Attention Cache Precision"
                                        value={kvPrecision}
                                        options={[
                                            { value: "model", label: kvDefaultLabel },
                                            { value: "16", label: "FP16/BF16 (16 bit)" },
                                            { value: "8", label: "FP8 (runtime support required)" },
                                            { value: "32", label: "FP32 (32 bit)" },
                                        ]}
                                        onChange={(value) => setKvPrecision(value as KvPrecisionChoice)}
                                    />
                                </div>
                            </div>
                        </section>
                        <section className="sizer-step" aria-labelledby="workload-step-title">
                            <StepHeading number="2" title="Workload" detail="Peak active demand" id="workload-step-title" />
                            <div className="grid-3">
                                <div className="field">
                                    <label>Input Tokens</label>
                                    <CustomSelect
                                        ariaLabel="Input Tokens"
                                        value={tokenPreset}
                                        options={[
                                            ...tokenPresets.map((value) => ({ value: String(value), label: `${value / 1024}K (${value.toLocaleString()})` })),
                                            { value: "custom", label: "Custom" },
                                        ]}
                                        onChange={changeTokenPreset}
                                    />
                                </div>
                                {tokenPreset === "custom" ? <TextField label="Custom Input Tokens" value={inputTokens} onChange={setInputTokens} numeric /> : null}
                                <TextField label="Concurrent Sequences" value={concurrent} onChange={setConcurrent} numeric />
                            </div>
                            <RangeField label="Growth Capacity" value={growthPercent} onChange={setGrowthPercent} min={0} max={100} suffix="%" />
                            <div className="effective-demand"><span>Effective concurrency after growth</span><strong>{Number.isFinite(effectiveConcurrent) ? effectiveConcurrent : "-"}</strong></div>
                            <span className="field-hint"><Info size={12} /> Active sequences, not idle users. Cache precision support depends on the serving runtime.</span>
                        </section>
                        <section className="sizer-step" aria-labelledby="hardware-step-title">
                            <StepHeading number="3" title="Hardware" detail="Deployment budget" id="hardware-step-title" />
                            <div className="button-row" role="group" aria-label="Hardware type">
                                <button
                                    type="button"
                                    className={`button ${!isDgxSpark ? "primary" : "secondary"}`}
                                    aria-pressed={!isDgxSpark}
                                    onClick={() => changeHardwareMode("gpu")}
                                >
                                    GPU
                                </button>
                                <button
                                    type="button"
                                    className={`button ${isDgxSpark ? "primary" : "secondary"}`}
                                    aria-pressed={isDgxSpark}
                                    onClick={() => changeHardwareMode("dgx_spark")}
                                >
                                    DGX Spark
                                </button>
                            </div>
                            <div className="grid-3">
                                {isDgxSpark ? (
                                    <div className="field">
                                        <label htmlFor="dgx-spark-count">DGX Spark Count</label>
                                        <select
                                            className="input"
                                            id="dgx-spark-count"
                                            value={gpuCount}
                                            onChange={(event) => setGpuCount(event.target.value)}
                                        >
                                            <option value="1">1 unit</option>
                                            <option value="2">2 units</option>
                                        </select>
                                    </div>
                                ) : <>
                                    <div className="field"><label>GPU Model</label><CustomSelect ariaLabel="GPU Model" value={gpuId} searchable options={catalog.gpus.filter((item) => item.id !== "dgx_spark").map((item) => ({ value: item.id, label: `${item.name} (${item.vramGb} GB)`, icon: <ProviderLogo name={item.vendor} /> }))} onChange={(id) => changeHardware(id, gpuCount)} /></div>
                                    <TextField label="GPU Count" value={gpuCount} onChange={(count) => changeHardware(gpuId, count)} numeric />
                                </>}
                                <TextField label="Runtime Peak (GB, estimated)" value={runtimeGb} onChange={setRuntimeGb} numeric />
                                {!isDgxSpark ? <>
                                    <TextField label="Host RAM (GB)" value={ram} onChange={(value) => { setRamDirty(true); setRam(value) }} numeric />
                                    <div className="field"><label>Host RAM Guideline</label><CustomSelect ariaLabel="Host RAM Guideline" value={ramMultiplier} options={[{ value: "2", label: "2 x total GPU memory" }, { value: "3", label: "3 x total GPU memory" }]} onChange={(value) => { setRamMultiplier(value); setRam(automaticRam(ram, ramDirty, gpu?.vramGb ?? 0, Number(gpuCount), Number(value))) }} /></div>
                                </> : null}
                            </div>
                            {!isDgxSpark ? <RangeField label="Target VRAM Utilization" value={targetUtilization} onChange={setTargetUtilization} min={1} max={100} suffix="%" /> : null}
                            <span className="field-hint"><Info size={12} /> {isDgxSpark ? "Uses the full 128 GB unified memory per unit. Two units are separate systems and are not a guaranteed shared pool." : "Host RAM guideline is 2-3x GPU memory. Multi-GPU requires an explicit partition plan."}</span>
                        </section>
                    </div>
                    {candidates.length ? (
                        <div className="panel page-stack recommendation-panel" style={{ gap: "0.75rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                                <h2 style={{ fontSize: "1rem" }}><Server size={16} /> Recommended GPU Configurations</h2>
                                <span className="field-hint"><Info size={12} /> Ranked by memory capacity</span>
                            </div>
                            <div className="card-grid recommendation-grid">
                                {candidates.map((item, i) => {
                                    const isSelected = gpuId === item.gpu.id && Number(gpuCount) === item.count
                                     const headroom = item.result?.breakdown.headroom
                                     const isFits = item.result?.memoryStatus === "fits"
                                     const doesNotFit = item.result?.memoryStatus === "does_not_fit"
                                     const needsPartition = item.result?.memoryStatus === "unknown"
                                         && item.result.missingInputs.length === 1
                                         && item.result.missingInputs[0].startsWith("Explicit per-device")
                                     const headroomGb = headroom != null ? headroom / GB : null

                                    return (
                                        <button
                                            key={`${item.gpu.id}-${item.count}`}
                                            type="button"
                                            className={`item-card ${isSelected ? "selected-card" : ""}`}
                                            style={{ display: "flex", flexDirection: "column", gap: "0.5rem", textAlign: "left", padding: "0.85rem" }}
                                            onClick={() => changeHardware(item.gpu.id, String(item.count))}
                                        >
                                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%", minWidth: 0, gap: "0.5rem" }}>
                                                <span className={`table-chip ${isFits ? (i === 0 ? "table-chip-blue" : "table-chip-gray") : "table-chip-gray"} inline-flex items-center shrink-0 leading-none px-2 py-0.5 text-[11px] font-medium h-[22px] gap-1`}>
                                                    {isFits ? (i === 0 ? <><Sparkles size={11} /> Minimum Fit</> : <><Layers size={11} /> Alternative Fit</>)
                                                        : needsPartition ? <><Server size={11} /> Multi-GPU Plan</>
                                                        : doesNotFit ? <><XCircle size={11} /> Insufficient VRAM</>
                                                        : <><AlertTriangle size={11} /> Needs Validation</>}
                                                </span>
                                                <StatusBadge
                                                    status={isFits ? "Fits" : needsPartition ? "Needs Partition" : doesNotFit ? "Does Not Fit" : item.error ? "Invalid Inputs" : "Unknown"}
                                                    size="sm"
                                                    iconOnly={!isFits}
                                                    tooltip={needsPartition ? "Multi-GPU inference requires an explicit tensor/pipeline partition plan" : item.error || undefined}
                                                />
                                            </div>
                                            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", margin: "0.2rem 0 0", minWidth: 0 }}>
                                                <ProviderLogo name={item.gpu.vendor} />
                                                <div style={{ minWidth: 0, flex: 1 }}>
                                                    <h3 style={{ margin: 0, fontSize: "0.925rem", lineHeight: 1.25, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={`${item.gpu.name} x ${item.count}`}>
                                                        {item.gpu.name}
                                                    </h3>
                                                    <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>Quantity: × {item.count}</span>
                                                </div>
                                            </div>
                                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", marginTop: "auto", width: "100%" }}>
                                                <span className="text-[var(--color-text-secondary)] font-mono">{item.gpu.vramGb * item.count} GB Total</span>
                                                {headroomGb != null ? (
                                                    <span className={`headroom-delta ${headroomGb >= 0 ? "surplus" : "deficit"}`} style={{ fontSize: "11px" }}>
                                                        {headroomGb >= 0 ? "+" : ""}{headroomGb.toFixed(1)} GB
                                                    </span>
                                                ) : null}
                                            </div>
                                        </button>
                                    )
                                })}
                            </div>
                        </div>
                    ) : null}
                </div>
                <div className="panel page-stack results-panel">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                        <h2><Activity size={18} /> Selected Hardware Estimate</h2>
                        <span className="field-hint"><Info size={12} /> Inference memory estimates only</span>
                    </div>
                    {live ? (
                        <SizingDetails
                            evaluation={live}
                            gpu={gpu}
                            gpuCount={gpuCount}
                            hostRamGb={ram}
                            targetUtilization={targetUtilization}
                            isDgxSpark={isDgxSpark}
                        />
                    ) : <p>Select a model.</p>}
                    <button type="button" id="confirm" disabled={saving || !live?.result} onClick={save}>{saving ? "Saving..." : "Save Sizing Snapshot"}</button>
                </div>
            </div>
            <Drawer title="Sizing History" open={historyOpen} onClose={() => setHistoryOpen(false)}>
                {history.map((item) => <div className="item-card" key={String(item.id)}><h3>{String(item.name)}</h3><StatusBadge status={item.calculationVersion === CALCULATION_VERSION ? "v3 snapshot" : "Legacy (old calculation)"} /><p>{new Date(String(item.createdAt)).toLocaleString()} | {String(item.hardwareType)}</p>{item.calculationVersion === CALCULATION_VERSION ? <details><summary>Saved inputs, metadata, policies and results</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{String(item.calculationSnapshot)}</pre></details> : <p className="muted">Historical result only; not recomputed or comparable to v3 memory estimates.</p>}</div>)}
                {!history.length ? <p>No saved history records.</p> : null}
            </Drawer>
        </section>
    )
}

function StepHeading({ number, title, detail, id }: { number: string, title: string, detail: string, id: string }) {
    return (
        <div className="sizer-step-heading">
            <div className="sizer-step-title">
                <span>{number}</span>
                <h2 id={id}>{title}</h2>
            </div>
            <span>{detail}</span>
        </div>
    )
}

function RangeField({ label, value, onChange, min, max, suffix }: { label: string, value: string, onChange: (value: string) => void, min: number, max: number, suffix: string }) {
    const id = useId()
    return (
        <div className="range-field">
            <div className="range-field-heading">
                <label htmlFor={id}>{label}</label>
                <div className="range-value">
                    <input aria-label={`${label} value`} type="number" min={min} max={max} value={value} onChange={(event) => onChange(event.target.value)} />
                    <span>{suffix}</span>
                </div>
            </div>
            <input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(event.target.value)} />
        </div>
    )
}

function TextField({ label, value, onChange, numeric = false }: { label: string, value: string, onChange: (value: string) => void, numeric?: boolean }) {
    const id = useId()
    return (
        <div className="field">
            <label className="field-label" htmlFor={id}>{label}</label>
            <input id={id} className="input" type={numeric ? "number" : "text"} min={numeric ? 0 : undefined} step={numeric ? "any" : undefined} value={value} onChange={(event) => onChange(event.target.value)} />
        </div>
    )
}
function SizingDetails({
    evaluation,
    gpu,
    gpuCount,
    hostRamGb,
    targetUtilization = "100",
    isDgxSpark = false,
}: {
    evaluation: Evaluation
    gpu?: Gpu
    gpuCount?: string
    hostRamGb?: string
    targetUtilization?: string
    isDgxSpark?: boolean
}) {
    const result = evaluation.result
    if (!result) {
        return (
            <div className="sizer-hero status-does_not_fit">
                <div className="sizer-hero-content">
                    <div className="sizer-hero-icon"><AlertCircle size={24} /></div>
                    <div className="sizer-hero-text">
                        <h3>Evaluation Error</h3>
                        <p>{evaluation.error}</p>
                    </div>
                </div>
            </div>
        )
    }

    const { breakdown, memoryStatus, perDevice, inputSnapshot, confidence, missingInputs, assumptions, evidence } = result
    const model = inputSnapshot.model
    const status = memoryStatus
    const isFit = status === "fits"
    const isOverflow = status === "does_not_fit"

    const workloadGb = breakdown.workload !== null ? (breakdown.workload / GB) : null
    const budgetGb = breakdown.usableBudget !== null ? (breakdown.usableBudget / GB) : null
    const headroomGb = breakdown.headroom !== null ? (breakdown.headroom / GB) : null
    const weightsGb = breakdown.weights !== null ? (breakdown.weights / GB) : null
    const kvGb = breakdown.kvCache !== null ? (breakdown.kvCache / GB) : null
    const runtimeGb = breakdown.runtimePeak !== null ? (breakdown.runtimePeak / GB) : null
    const capacity = !isDgxSpark && gpu && breakdown.workload !== null
        ? capacityEstimate(breakdown.workload, gpu, targetUtilization)
        : null
    const selectedGpuCount = Number(gpuCount ?? 1)

    const totalRef = Math.max(budgetGb ?? 0, workloadGb ?? 0, 1)
    const weightsPct = weightsGb !== null ? Math.min((weightsGb / totalRef) * 100, 100) : 0
    const kvPct = kvGb !== null ? Math.min((kvGb / totalRef) * 100, 100) : 0
    const runtimePct = runtimeGb !== null ? Math.min((runtimeGb / totalRef) * 100, 100) : 0
    const freePct = headroomGb !== null && headroomGb > 0 ? Math.min((headroomGb / totalRef) * 100, 100) : 0
    const overflowPct = headroomGb !== null && headroomGb < 0 ? Math.min((Math.abs(headroomGb) / totalRef) * 100, 100) : 0

    return (
        <div className="page-stack">
            {/* Hero Status Banner */}
            <div className={`sizer-hero status-${status}`}>
                <div className="sizer-hero-content">
                    <div className="sizer-hero-icon">
                        {isFit ? <CheckCircle2 size={26} /> : isOverflow ? <XCircle size={26} /> : <AlertTriangle size={26} />}
                    </div>
                    <div className="sizer-hero-text">
                        <h3>
                            {isFit ? "Memory Fit (Estimated)" : isOverflow ? "Does Not Fit (Overflow)" : "Needs Validation / Partition"}
                        </h3>
                        <p>
                            {isFit
                                ? `Inference workload fits in memory with ${headroomGb?.toFixed(2)} GB free headroom.`
                                : isOverflow
                                ? `Workload exceeds budget by ${headroomGb !== null ? Math.abs(headroomGb).toFixed(2) : "-"} GB. More VRAM needed.`
                                : isDgxSpark && selectedGpuCount === 2 ? "Two separate systems; unified memory is not a guaranteed shared pool. Partitioning is unknown." : "Multi-GPU inference requires an explicit tensor/pipeline partition plan."}
                        </p>
                    </div>
                </div>
                <div className="sizer-hero-tags">
                    <span className="table-chip table-chip-blue">
                        <Cpu size={12} /> {model.totalParameters ? `${(model.totalParameters / 1e9).toFixed(2)}B` : "Weights"} ({inputSnapshot.weightFormat})
                    </span>
                    <span className={`table-chip ${metadataStatus(model) === "ready" ? "table-chip-green" : "table-chip-yellow"}`}>
                        <ShieldCheck size={12} /> {metadataStatus(model)}
                    </span>
                    <span className="table-chip table-chip-gray">
                        <Sparkles size={12} /> {confidence}
                    </span>
                </div>
            </div>

            {gpu ? (
                <div className="server-summary-bar hardware-summary" aria-label="Hardware summary">
                    <span className="hardware-summary-device">
                        <ProviderLogo name={gpu.vendor} size="xs" />
                        {gpu.name} × {selectedGpuCount}
                    </span>
                    <span className="hardware-summary-metric">
                        <span>{isDgxSpark ? "Unified memory" : "Total VRAM"}</span>
                        <span>{isDgxSpark ? `${gpu.vramGb} GB per unit` : `${gpu.vramGb * selectedGpuCount} GB`}</span>
                    </span>
                    {!isDgxSpark ? (
                        <span className="hardware-summary-metric">
                            <span>Host RAM</span>
                            <span>{hostRamGb ? `${Number(hostRamGb).toFixed(0)} GB` : "Not set"}</span>
                        </span>
                    ) : null}
                    {!isDgxSpark ? (
                        <span className="hardware-summary-metric">
                            <span>Target utilization</span>
                            <span>{targetUtilization}%</span>
                        </span>
                    ) : (
                        <span className="hardware-summary-metric">
                            <span>Full memory budget per unit</span>
                            <span>128 GB</span>
                        </span>
                    )}
                </div>
            ) : null}

            {/* Evidence-safe result cards. Capacity floor does not certify a multi-GPU partition. */}
            <div className="sizer-stats-grid">
                <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><Layers size={13} /> Weight Memory</span>
                    <span className="sizer-stat-value">{weightsGb !== null ? `${weightsGb.toFixed(1)} GB` : "Unknown"}</span>
                </div>
                <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><HardDrive size={13} /> Attention Cache</span>
                    <span className="sizer-stat-value">{kvGb !== null ? `${kvGb.toFixed(1)} GB` : "Unknown"}</span>
                </div>
                <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><Zap size={13} /> Runtime Peak</span>
                    <span className="sizer-stat-value">{runtimeGb !== null ? `${runtimeGb.toFixed(1)} GB` : "Unknown"}</span>
                </div>
                <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><Activity size={13} /> Total Estimated</span>
                    <span className="sizer-stat-value">{workloadGb !== null ? `${workloadGb.toFixed(1)} GB` : "Unknown"}</span>
                </div>
                <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><Cpu size={13} /> Required VRAM @ {targetUtilization}%</span>
                    <span className="sizer-stat-value">{capacity ? `${(capacity.requiredPhysicalBytes / GB).toFixed(1)} GB` : "Unknown"}</span>
                </div>
                {!isDgxSpark ? <div className="sizer-stat-card">
                    <span className="sizer-stat-label"><Activity size={13} /> Capacity Utilization</span>
                    <span className="sizer-stat-value">{capacity ? `${capacity.utilizationPercent.toFixed(1)}%` : "Unknown"}</span>
                </div> : null}
            </div>

            {/* Visual Memory Allocation Bar */}
            <div className="sizer-meter-container">
                <div className="sizer-meter-header">
                    <span>Memory Allocation Breakdown</span>
                    <span className="text-xs font-mono text-[var(--color-text-secondary)]">
                        {workloadGb !== null ? `${workloadGb.toFixed(1)} GB` : "0 GB"} / {budgetGb !== null ? `${budgetGb.toFixed(1)} GB` : "0 GB"} ({budgetGb && workloadGb ? Math.round((workloadGb / budgetGb) * 100) : 0}%)
                    </span>
                </div>
                <div className="sizer-meter-bar" title="Memory Allocation Visualizer">
                    <div className="sizer-meter-segment weights" style={{ width: `${weightsPct}%` }} title={`Weights: ${weightsGb?.toFixed(1)} GB`} />
                    <div className="sizer-meter-segment kv" style={{ width: `${kvPct}%` }} title={`Attention Cache: ${kvGb?.toFixed(1)} GB`} />
                    <div className="sizer-meter-segment runtime" style={{ width: `${runtimePct}%` }} title={`Runtime Peak: ${runtimeGb?.toFixed(1)} GB`} />
                    {freePct > 0 ? (
                        <div className="sizer-meter-segment headroom" style={{ width: `${freePct}%` }} title={`Free Headroom: ${headroomGb?.toFixed(1)} GB`} />
                    ) : null}
                    {overflowPct > 0 ? (
                        <div className="sizer-meter-segment overflow" style={{ width: `${overflowPct}%` }} title={`Overflow: ${headroomGb?.toFixed(1)} GB`} />
                    ) : null}
                </div>
                <div className="sizer-meter-legend">
                    <span className="sizer-legend-item">
                        <span className="sizer-legend-dot weights" /> Weights: <strong className="font-mono">{weightsGb?.toFixed(1) ?? "-"} GB</strong>
                    </span>
                    <span className="sizer-legend-item">
                        <span className="sizer-legend-dot kv" /> Attention Cache: <strong className="font-mono">{kvGb?.toFixed(1) ?? "-"} GB</strong>
                    </span>
                    <span className="sizer-legend-item">
                        <span className="sizer-legend-dot runtime" /> Runtime: <strong className="font-mono">{runtimeGb?.toFixed(1) ?? "-"} GB</strong>
                    </span>
                    {headroomGb !== null && headroomGb >= 0 ? (
                        <span className="sizer-legend-item">
                            <span className="sizer-legend-dot headroom" /> Free: <strong className="font-mono text-emerald-600 dark:text-emerald-400">+{headroomGb.toFixed(1)} GB</strong>
                        </span>
                    ) : headroomGb !== null ? (
                        <span className="sizer-legend-item">
                            <span className="sizer-legend-dot overflow" /> Deficit: <strong className="font-mono text-red-600 dark:text-red-400">{headroomGb.toFixed(1)} GB</strong>
                        </span>
                    ) : null}
                </div>
            </div>

            {/* Per Device Breakdown (Multi-GPU) */}
            {perDevice.length > 1 ? (
                <div className="page-stack" style={{ gap: "0.5rem" }}>
                    <h3 style={{ fontSize: "0.85rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.02em", color: "var(--color-text-secondary)" }}>
                        Device Placements ({perDevice.length} GPUs)
                    </h3>
                    <div className="grid-2" style={{ gap: "0.5rem" }}>
                        {perDevice.map((device) => {
                            const devWorkload = (device.workloadBytes / GB).toFixed(1)
                            const devBudget = (device.budgetBytes / GB).toFixed(1)
                            const devHeadroom = (device.headroomBytes / GB).toFixed(1)
                            const isDevFit = device.headroomBytes >= 0
                            return (
                                <div className="sizer-stat-card" key={device.id} style={{ padding: "0.6rem 0.75rem" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                        <span style={{ fontWeight: 600, fontSize: "0.8rem", display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                                            {gpu ? <ProviderLogo name={gpu.vendor} size="xs" /> : <Server size={14} />} {device.node} / {device.id}
                                        </span>
                                        <span className={`table-chip ${isDevFit ? "table-chip-green" : "table-chip-red"}`} style={{ padding: "1px 6px", fontSize: "10px" }}>
                                            {isDevFit ? `+${devHeadroom} GB` : `${devHeadroom} GB`}
                                        </span>
                                    </div>
                                    <span className="text-xs text-[var(--color-text-secondary)] font-mono">
                                        {devWorkload} GB / {devBudget} GB budget
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            ) : null}

            {/* Missing Inputs Alert */}
            {missingInputs.length ? (
                <div className="table-chip table-chip-yellow" style={{ padding: "0.6rem 0.85rem", borderRadius: "var(--radius-min)", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "0.35rem", width: "100%" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontWeight: 600 }}>
                        <AlertTriangle size={14} /> Missing Required Metadata:
                    </div>
                    <ul style={{ margin: "0 0 0 1.25rem", padding: 0, fontSize: "0.775rem" }}>
                        {missingInputs.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                </div>
            ) : null}

            {/* Collapsible Assumptions & Reference Notes */}
            <details className="sizer-accordion">
                <summary><Info size={14} /> Technical Assumptions & Operational Provenance</summary>
                <div className="sizer-accordion-content page-stack" style={{ gap: "0.6rem" }}>
                    <div>
                        <strong>Storage Heuristic:</strong>{" "}
                        {breakdown.weights === null ? "Unknown" : `${((breakdown.weights * 2 + 200 * GB) / GB).toFixed(1)} GB`} (covers 2 weight copies + 200 GB operational workspace).
                    </div>
                    <div>
                        <strong>Assumptions:</strong>
                        <ul style={{ margin: "0.25rem 0 0 1.25rem", padding: 0 }}>
                            {assumptions.map((item) => <li key={item}>{item}</li>)}
                        </ul>
                    </div>
                    {breakdown.fullAttentionKv !== null ? (
                        <div>
                            <strong>Attention Cache:</strong>
                            <ul style={{ margin: "0.25rem 0 0 1.25rem", padding: 0 }}>
                                <li>Full attention: {(breakdown.fullAttentionKv / GB).toFixed(2)} GB</li>
                                {breakdown.slidingAttentionKv > 0 ? <li>Sliding-window attention: {(breakdown.slidingAttentionKv / GB).toFixed(2)} GB</li> : null}
                                {breakdown.linearRecurrentState > 0 ? <li>Linear recurrent state: {(breakdown.linearRecurrentState / GB).toFixed(2)} GB</li> : null}
                                {breakdown.linearConvState > 0 ? <li>Linear convolution state: {(breakdown.linearConvState / GB).toFixed(2)} GB</li> : null}
                            </ul>
                        </div>
                    ) : null}
                    {evidence.length ? (
                        <div>
                            <strong>Evidence Sources:</strong>
                            <div style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap", marginTop: "0.25rem" }}>
                                {evidence.map((item) => (
                                    <a key={item} href={item} target="_blank" rel="noreferrer" className="table-chip table-chip-gray hover:text-blue-500" style={{ wordBreak: "break-all" }}>
                                        {item}
                                    </a>
                                ))}
                            </div>
                        </div>
                    ) : null}
                </div>
            </details>
        </div>
    )
}
function isLlmModel(model: Model) {
    const type = model.modelType?.toLowerCase()
    return !type || ["llm", "chat", "coding", "agentic"].includes(type)
}

function isMemoryReady(model: Model) {
    return metadataStatus(model) === "ready"
}

function metadataStatus(model: ModelMetadata) {
    if (!model.totalParameters || !model.numHiddenLayers || !model.numAttentionHeads || !model.numKeyValueHeads || (!model.headDim && !model.hiddenSize)) return "incomplete"
    const fullAttention = ["decoder_mha", "decoder_gqa", "decoder_mqa"].includes(model.architecture ?? "")
    const profiledAttention = ["sliding_window", "hybrid"].includes(model.architecture ?? "") && Boolean(model.attentionProfile)
    if (!fullAttention && !profiledAttention) return "unsupported architecture"
    return "ready"
}
