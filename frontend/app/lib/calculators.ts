export const CALCULATION_VERSION = 3
export const GB = 1_000_000_000
export const GiB = 1_073_741_824
export const WEIGHT_BITS = { FP16: 16, BF16: 16, INT8: 8, FP8: 8, INT4: 4, FP4: 4 } as const
export type WeightPrecision = keyof typeof WEIGHT_BITS
export type WeightPrecisionChoice = WeightPrecision | "model"
export type KvPrecisionChoice = "model" | "8" | "16" | "32"
export const ARCHITECTURES = ["decoder_mha", "decoder_gqa", "decoder_mqa", "mla", "sliding_window", "hybrid", "encoder", "multimodal"] as const
export const metadataNumbers = ["totalParameters", "activeParameters", "numHiddenLayers", "numAttentionHeads", "numKeyValueHeads", "headDim", "hiddenSize", "maxContextTokens", "weightBits", "kvCacheBits", "residentWeightBytes"] as const
export const metadataStrings = ["modelId", "revision", "architecture", "attentionProfile", "weightFormat", "metadataSource", "verifiedAt", "residentWeightEngine", "residentWeightEngineVersion", "residentWeightSource"] as const
export type ModelMetadata = Partial<Record<typeof metadataNumbers[number], number | null> & Record<typeof metadataStrings[number], string | null>>

export function resolveWeightPrecision(model: ModelMetadata, choice: WeightPrecisionChoice) {
    if (choice !== "model") return { format: choice, bits: WEIGHT_BITS[choice], source: "manual" as const }
    if (model.weightFormat && model.weightBits && model.weightBits <= 32) {
        return { format: model.weightFormat, bits: model.weightBits, source: "model" as const }
    }
    return { format: "FP16", bits: WEIGHT_BITS.FP16, source: "fallback" as const }
}

export function resolveKvCacheBits(model: ModelMetadata, choice: KvPrecisionChoice) {
    if (choice !== "model") return { bits: Number(choice), source: "manual" as const }
    if (model.kvCacheBits && [8, 16, 32].includes(model.kvCacheBits)) {
        return { bits: model.kvCacheBits, source: "model" as const }
    }
    return { bits: 16, source: "fallback" as const }
}

type SlidingAttentionProfile = {
    kind: "sliding_full"
    fullLayers: number
    slidingLayers: number
    slidingWindow: number
}

type LinearAttentionProfile = {
    kind: "linear_full"
    fullLayers: number
    linearLayers: number
    linear: {
        convKernelDim: number
        keyHeadDim: number
        valueHeadDim: number
        numKeyHeads: number
        numValueHeads: number
        convStateBits: number
        recurrentStateBits: number
    }
}

export type AttentionProfile = SlidingAttentionProfile | LinearAttentionProfile

function positiveInteger(value: unknown, name: string) {
    number(value, name, 1, true)
    return value
}

export function parseAttentionProfile(value: string | null | undefined): AttentionProfile | null {
    if (!value) return null
    let profile: unknown
    try {
        profile = JSON.parse(value)
    } catch {
        throw new Error("attentionProfile must be valid JSON")
    }
    if (!profile || typeof profile !== "object") throw new Error("attentionProfile must be an object")
    const data = profile as Record<string, unknown>
    if (data.kind === "sliding_full") {
        return {
            kind: data.kind,
            fullLayers: positiveInteger(data.fullLayers, "fullLayers"),
            slidingLayers: positiveInteger(data.slidingLayers, "slidingLayers"),
            slidingWindow: positiveInteger(data.slidingWindow, "slidingWindow"),
        }
    }
    if (data.kind === "linear_full") {
        if (!data.linear || typeof data.linear !== "object") throw new Error("linear attention dimensions are required")
        const linear = data.linear as Record<string, unknown>
        return {
            kind: data.kind,
            fullLayers: positiveInteger(data.fullLayers, "fullLayers"),
            linearLayers: positiveInteger(data.linearLayers, "linearLayers"),
            linear: {
                convKernelDim: positiveInteger(linear.convKernelDim, "convKernelDim"),
                keyHeadDim: positiveInteger(linear.keyHeadDim, "keyHeadDim"),
                valueHeadDim: positiveInteger(linear.valueHeadDim, "valueHeadDim"),
                numKeyHeads: positiveInteger(linear.numKeyHeads, "numKeyHeads"),
                numValueHeads: positiveInteger(linear.numValueHeads, "numValueHeads"),
                convStateBits: positiveInteger(linear.convStateBits, "convStateBits"),
                recurrentStateBits: positiveInteger(linear.recurrentStateBits, "recurrentStateBits"),
            },
        }
    }
    throw new Error("Unsupported attentionProfile kind")
}

function number(value: unknown, name: string, min = 0, integer = false): asserts value is number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < min || (integer && !Number.isSafeInteger(value))) {
        throw new Error(`${name} must be a finite ${integer ? "integer " : ""}number >= ${min}`)
    }
}

// Also used by CRUD with the merged record, so partial updates cannot bypass cross-field checks.
export function validateModelMetadata(model: ModelMetadata) {
    for (const key of metadataNumbers) {
        if (model[key] != null) number(model[key], key, 1, !["weightBits", "kvCacheBits"].includes(key))
    }
    for (const key of metadataStrings) {
        if (model[key] != null && (typeof model[key] !== "string" || !model[key]!.trim())) throw new Error(`${key} must be nonempty text or null`)
    }
    if (model.architecture && !ARCHITECTURES.includes(model.architecture as typeof ARCHITECTURES[number])) throw new Error("Unsupported architecture name")
    const attentionProfile = parseAttentionProfile(model.attentionProfile)
    if (attentionProfile && model.numHiddenLayers) {
        const profileLayers = attentionProfile.kind === "sliding_full"
            ? attentionProfile.fullLayers + attentionProfile.slidingLayers
            : attentionProfile.fullLayers + attentionProfile.linearLayers
        if (profileLayers !== model.numHiddenLayers) throw new Error("attentionProfile layers must match numHiddenLayers")
    }
    if (attentionProfile?.kind === "sliding_full" && model.architecture !== "sliding_window") throw new Error("sliding_full profile requires sliding_window architecture")
    if (attentionProfile?.kind === "linear_full" && model.architecture !== "hybrid") throw new Error("linear_full profile requires hybrid architecture")
    if (model.verifiedAt && !/^\d{4}-\d{2}-\d{2}$/.test(model.verifiedAt)) throw new Error("verifiedAt must be YYYY-MM-DD")
    if (model.verifiedAt && (!Number.isFinite(Date.parse(model.verifiedAt)) || new Date(model.verifiedAt).toISOString().slice(0, 10) !== model.verifiedAt)) throw new Error("Invalid verifiedAt")
    for (const key of ["metadataSource", "residentWeightSource"] as const) {
        if (model[key]) {
            let url: URL
            try { url = new URL(model[key]!) } catch { throw new Error(`${key} must be an HTTP(S) URL`) }
            if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password) throw new Error(`${key} must be an HTTP(S) URL without credentials`)
        }
    }
    if (model.activeParameters && model.totalParameters && model.activeParameters > model.totalParameters) throw new Error("activeParameters exceeds totalParameters")
    const q = model.numAttentionHeads, kv = model.numKeyValueHeads
    if (q && kv && (kv > q || q % kv !== 0)) throw new Error("KV heads must divide query heads")
    if (model.architecture === "decoder_mha" && q && kv && q !== kv) throw new Error("MHA requires equal query and KV heads")
    if (model.architecture === "decoder_mqa" && kv && kv !== 1) throw new Error("MQA requires one KV head")
    if (model.weightBits && model.weightBits > 32) throw new Error("weightBits must be <= 32")
    if (model.kvCacheBits && ![8, 16, 32].includes(model.kvCacheBits)) throw new Error("KV bits must be 8, 16, or 32")
    if (model.residentWeightBytes && ["modelId", "revision", "weightFormat", "weightBits", "residentWeightEngine", "residentWeightEngineVersion", "residentWeightSource", "verifiedAt"].some((key) => !model[key as keyof ModelMetadata])) {
        throw new Error("Measured resident weights require model/revision, format/bits, engine/version, source and verification date")
    }
}

export type Device = { id: string, node: string, physicalMemoryBytes: number, systemReservedBytes: number, safetyReservedBytes: number }
export type Partition = {
    tensorParallelSize: number
    pipelineParallelSize: number
    replicaCount: number
    source: string
    replicas: { concurrentSequences: number, placements: { deviceId: string, weightFraction: number, kvFraction: number, runtimePeakBytes: number }[] }[]
}
export type SizingInput = {
    model: ModelMetadata
    weightFormat: string
    weightBits: number
    kvCacheBits: number
    inputTokens: number
    maxOutputTokens: number
    concurrentSequences: number
    servingEngine: string
    engineVersion: string
    runtimePeakBytes: number
    quantizationOverheadPercent: number
    devices: Device[]
    partition?: Partition | null
    targetTTFTMs?: number | null
    targetTPOTMs?: number | null
    targetOutputTokensPerSecondPerRequest?: number | null
}
export type MemoryStatus = "fits" | "does_not_fit" | "unknown"
export const MEMORY_LABELS: Record<MemoryStatus, string> = { fits: "Memory fit (estimated)", does_not_fit: "Does not fit this configuration", unknown: "Unknown" }

export function calculateInference(input: SizingInput) {
    validateModelMetadata(input.model)
    for (const key of ["inputTokens", "maxOutputTokens"] as const) number(input[key], key, 0, true)
    number(input.concurrentSequences, "concurrentSequences", 1, true)
    number(input.weightBits, "weightBits", 1)
    if (input.weightBits > 32 || ![8, 16, 32].includes(input.kvCacheBits)) throw new Error("Invalid precision")
    if (!input.weightFormat.trim()) throw new Error("Weight format is required")
    number(input.runtimePeakBytes, "runtimePeakBytes")
    number(input.quantizationOverheadPercent, "quantizationOverheadPercent")
    for (const key of ["targetTTFTMs", "targetTPOTMs", "targetOutputTokensPerSecondPerRequest"] as const) if (input[key] != null) number(input[key], key, Number.MIN_VALUE)
    const tokens = input.inputTokens + input.maxOutputTokens
    number(tokens, "tokens per sequence", 1, true)
    if (input.model.maxContextTokens && tokens > input.model.maxContextTokens) throw new Error("Input + output exceeds verified context limit")
    if (!input.devices.length) throw new Error("At least one device/node is required")
    const ids = new Set<string>()
    const budgets = input.devices.map((device) => {
        if (!device.id || !device.node || ids.has(device.id)) throw new Error("Unique device IDs and node placement are required")
        ids.add(device.id)
        number(device.physicalMemoryBytes, "physicalMemoryBytes", 1)
        number(device.systemReservedBytes, "systemReservedBytes")
        number(device.safetyReservedBytes, "safetyReservedBytes")
        const budget = device.physicalMemoryBytes - device.systemReservedBytes - device.safetyReservedBytes
        if (budget < 0) throw new Error("Reserves exceed physical memory")
        return { ...device, usableBudgetBytes: budget }
    })
    const missingInputs: string[] = []
    const assumptions = ["GPU-resident inference only; no offload, training or fine-tuning.", "Attention cache uses the verified layer profile where available; no prefix sharing. Physical block rounding requires a runtime profile.", "Runtime peak is a user estimate covering buffers, activations and workspace, not a benchmark.", "Catalog memory is interpreted as decimal GB; verify actual device capacity before procurement."]
    const model = input.model
    const rawWeights = model.totalParameters ? model.totalParameters * input.weightBits / 8 : null
    if (rawWeights === null) missingInputs.push("totalParameters (resident total, not active MoE parameters)")
    const measured = model.residentWeightBytes != null && model.weightBits === input.weightBits && model.weightFormat === input.weightFormat && model.residentWeightEngine === input.servingEngine && model.residentWeightEngineVersion === input.engineVersion
    const quantizationOverhead = rawWeights === null ? null : measured ? 0 : rawWeights * input.quantizationOverheadPercent / 100
    const weights = measured ? model.residentWeightBytes! : rawWeights === null ? null : rawWeights + quantizationOverhead!
    assumptions.push(measured ? "Matching measured resident weights used; no added quantization overhead." : `Raw weights + ${input.quantizationOverheadPercent}% estimated overhead. Artifact file size is not resident memory.`)
    const attentionProfile = parseAttentionProfile(model.attentionProfile)
    const fullAttention = ["decoder_mha", "decoder_gqa", "decoder_mqa"].includes(model.architecture ?? "")
    const slidingAttention = model.architecture === "sliding_window" && attentionProfile?.kind === "sliding_full"
    const linearAttention = model.architecture === "hybrid" && attentionProfile?.kind === "linear_full"
    const supported = fullAttention || slidingAttention || linearAttention
    if (!supported) missingInputs.push("Supported attention architecture/profile (full, sliding/full, or linear/full)")
    const headDim = model.headDim ?? (supported && model.hiddenSize && model.numAttentionHeads && model.hiddenSize % model.numAttentionHeads === 0 ? model.hiddenSize / model.numAttentionHeads : null)
    for (const key of ["numHiddenLayers", "numAttentionHeads", "numKeyValueHeads"] as const) if (!model[key]) missingInputs.push(key)
    if (!headDim) missingInputs.push("headDim (or divisible hiddenSize/query heads for supported architecture)")
    const cacheGeometryReady = supported && model.numHiddenLayers && model.numAttentionHeads && model.numKeyValueHeads && headDim
    const bytesPerKvElement = input.kvCacheBits / 8
    const kvPerLayerToken = cacheGeometryReady ? 2 * model.numKeyValueHeads! * headDim! * bytesPerKvElement : null
    let fullAttentionKv: number | null = null
    let slidingAttentionKv = 0
    let linearRecurrentState = 0
    let linearConvState = 0
    if (kvPerLayerToken !== null) {
        const fullLayers = fullAttention ? model.numHiddenLayers! : attentionProfile!.fullLayers
        fullAttentionKv = kvPerLayerToken * fullLayers * tokens * input.concurrentSequences
        if (attentionProfile?.kind === "sliding_full") {
            slidingAttentionKv = kvPerLayerToken
                * attentionProfile.slidingLayers
                * Math.min(tokens, attentionProfile.slidingWindow)
                * input.concurrentSequences
            assumptions.push(`${attentionProfile.fullLayers} full-attention layers use full context; ${attentionProfile.slidingLayers} sliding layers cache at most ${attentionProfile.slidingWindow} tokens.`)
        }
        if (attentionProfile?.kind === "linear_full") {
            const linear = attentionProfile.linear
            linearRecurrentState = attentionProfile.linearLayers
                * input.concurrentSequences
                * linear.numValueHeads
                * linear.keyHeadDim
                * linear.valueHeadDim
                * linear.recurrentStateBits / 8
            const convElements = 2 * linear.numKeyHeads * linear.keyHeadDim
                + linear.numValueHeads * linear.valueHeadDim
            linearConvState = attentionProfile.linearLayers
                * input.concurrentSequences
                * convElements
                * linear.convKernelDim
                * linear.convStateBits / 8
            assumptions.push(`${attentionProfile.fullLayers} full-attention layers use token KV; ${attentionProfile.linearLayers} linear layers use fixed recurrent and convolution states from the pinned Transformers profile.`)
        }
    }
    const kvCache = fullAttentionKv === null
        ? null
        : fullAttentionKv + slidingAttentionKv + linearRecurrentState + linearConvState
    const attentionCachePerSequence = kvCache === null ? null : kvCache / input.concurrentSequences
    const workload = weights === null || kvCache === null ? null : weights + kvCache + input.runtimePeakBytes
    const physical = budgets.reduce((sum, d) => sum + d.physicalMemoryBytes, 0)
    const budget = budgets.reduce((sum, d) => sum + d.usableBudgetBytes, 0)
    let perDevice: { id: string, node: string, workloadBytes: number, budgetBytes: number, headroomBytes: number }[] = []
    let allocatedWeights: number | null = weights, allocatedKv: number | null = kvCache, allocatedRuntime = input.runtimePeakBytes
    if (input.partition) {
        const p = input.partition
        for (const key of ["tensorParallelSize", "pipelineParallelSize", "replicaCount"] as const) number(p[key], key, 1, true)
        if (!p.source?.trim() || !input.servingEngine.trim() || !input.engineVersion.trim()) throw new Error("Partition requires source and serving engine/version")
        if (!Array.isArray(p.replicas) || p.replicas.length !== p.replicaCount) throw new Error("Replica count does not match plan")
        let sequences = 0
        const used = new Set<string>()
        const allocations = new Map<string, { weights: number, kv: number, runtime: number }>()
        for (const replica of p.replicas) {
            number(replica.concurrentSequences, "replica concurrentSequences", 0, true)
            sequences += replica.concurrentSequences
            if (!Array.isArray(replica.placements) || replica.placements.length !== p.tensorParallelSize * p.pipelineParallelSize) throw new Error("Placements must match TP x PP per replica")
            let weightCoverage = 0, kvCoverage = 0
            for (const placement of replica.placements) {
                if (!ids.has(placement.deviceId) || used.has(placement.deviceId)) throw new Error("Each placement must use a distinct configured device")
                used.add(placement.deviceId)
                number(placement.weightFraction, "weightFraction")
                number(placement.kvFraction, "kvFraction")
                number(placement.runtimePeakBytes, "placement runtimePeakBytes")
                if (placement.weightFraction > 1 || placement.kvFraction > 1) throw new Error("Shard fractions must be <= 1")
                weightCoverage += placement.weightFraction
                kvCoverage += placement.kvFraction
                allocations.set(placement.deviceId, { weights: (weights ?? 0) * placement.weightFraction, kv: (attentionCachePerSequence ?? 0) * replica.concurrentSequences * placement.kvFraction, runtime: placement.runtimePeakBytes })
            }
            if (weightCoverage < 1 - 1e-10 || kvCoverage < 1 - 1e-10) throw new Error("Each replica must contain full weights and KV coverage; replication may exceed 1")
        }
        if (sequences !== input.concurrentSequences || used.size !== ids.size) throw new Error("Plan must cover every sequence and configured device")
        allocatedRuntime = [...allocations.values()].reduce((sum, a) => sum + a.runtime, 0)
        allocatedWeights = weights === null ? null : [...allocations.values()].reduce((sum, a) => sum + a.weights, 0)
        allocatedKv = kvCache === null ? null : [...allocations.values()].reduce((sum, a) => sum + a.kv, 0)
        if (workload !== null) perDevice = budgets.map((device) => {
            const a = allocations.get(device.id)!
            const bytes = a.weights + a.kv + a.runtime
            return { id: device.id, node: device.node, workloadBytes: bytes, budgetBytes: device.usableBudgetBytes, headroomBytes: device.usableBudgetBytes - bytes }
        })
        assumptions.push("Explicit user partition: runtime bytes per placement replace the single-device runtime estimate; fractions include KV replication. Compatibility is not verified by a placement plan.")
    } else if (budgets.length === 1 && workload !== null) {
        perDevice = [{ id: budgets[0].id, node: budgets[0].node, workloadBytes: workload, budgetBytes: budget, headroomBytes: budget - workload }]
    } else if (budgets.length > 1) missingInputs.push("Explicit per-device/per-node partition plan, KV replication and replica isolation")
    const allocatedWorkload = allocatedWeights === null || allocatedKv === null ? null : allocatedWeights + allocatedKv + allocatedRuntime
    const memoryStatus: MemoryStatus = perDevice.length ? perDevice.some((d) => d.headroomBytes < 0) ? "does_not_fit" : "fits" : "unknown"
    if (!model.modelId || !model.revision || !model.metadataSource || !model.verifiedAt) assumptions.push("Model identity/revision or metadata provenance is incomplete; verify catalog inputs.")
    if (!model.maxContextTokens) assumptions.push("Model/runtime context limit is unverified.")
    for (const value of [rawWeights, weights, quantizationOverhead, kvCache, allocatedWeights, allocatedKv, allocatedWorkload, physical, budget]) if (value !== null) number(value, "calculated bytes")
    return {
        calculationVersion: CALCULATION_VERSION,
        inputSnapshot: input,
        memoryStatus,
        compatibilityStatus: "unknown" as const,
        performanceStatus: "not_benchmarked" as const,
        supportedConcurrent: null,
        confidence: missingInputs.length ? "incomplete" as const : "estimated" as const,
        breakdown: { rawWeights, quantizationOverhead, weights, kvCache, fullAttentionKv, slidingAttentionKv, linearRecurrentState, linearConvState, runtimePeak: allocatedRuntime, allocatedWeights, allocatedKv, workload: allocatedWorkload, physicalMemory: physical, usableBudget: budget, reserve: physical - budget, headroom: perDevice.length && allocatedWorkload !== null ? budget - allocatedWorkload : null },
        perDevice,
        assumptions,
        missingInputs,
        evidence: [model.metadataSource, measured ? model.residentWeightSource : null, input.partition?.source].filter((s): s is string => Boolean(s)),
    }
}

export type HardwareInput = { memoryGb: number, deviceCount: number, systemReservedGb: number, safetyReservePercent: number }
export function standardDevices(hardware: HardwareInput): Device[] {
    number(hardware.deviceCount, "deviceCount", 1, true)
    if (hardware.deviceCount > 64) throw new Error("At most 64 devices supported")
    number(hardware.memoryGb, "memoryGb", 1)
    number(hardware.systemReservedGb, "systemReservedGb")
    number(hardware.safetyReservePercent, "safetyReservePercent")
    return Array.from({ length: hardware.deviceCount }, (_, i) => ({ id: `device-${i + 1}`, node: "node-1", physicalMemoryBytes: hardware.memoryGb * GB, systemReservedBytes: hardware.systemReservedGb * GB, safetyReservedBytes: hardware.memoryGb * GB * hardware.safetyReservePercent / 100 }))
}
export function calculateStandardGpu(input: Omit<SizingInput, "devices"> & { hardware: HardwareInput }) {
    return calculateInference({ ...input, devices: standardDevices(input.hardware) })
}
export function automaticRam(current: string, dirty: boolean, memoryGb: number, count: number, multiplier: number) {
    return dirty ? current : String(memoryGb * count * multiplier)
}
export function calculateCapacityFloor(
    workloadBytes: number,
    memoryGb: number,
    targetUtilizationPercent: number,
    systemReservedGb = 0,
) {
    number(workloadBytes, "workloadBytes")
    number(memoryGb, "memoryGb", Number.MIN_VALUE)
    number(targetUtilizationPercent, "targetUtilizationPercent", Number.MIN_VALUE)
    number(systemReservedGb, "systemReservedGb")
    if (targetUtilizationPercent > 100) throw new Error("targetUtilizationPercent must be <= 100")
    const physicalPerDevice = memoryGb * GB
    const usablePerDevice = physicalPerDevice * targetUtilizationPercent / 100 - systemReservedGb * GB
    if (usablePerDevice <= 0) throw new Error("Target utilization and system reserve leave no usable memory")
    const deviceCount = Math.max(1, Math.ceil(workloadBytes / usablePerDevice))
    const installedPhysicalBytes = physicalPerDevice * deviceCount
    const requiredPhysicalBytes = (workloadBytes + systemReservedGb * GB * deviceCount) / (targetUtilizationPercent / 100)
    return {
        deviceCount,
        requiredPhysicalBytes,
        installedPhysicalBytes,
        utilizationPercent: workloadBytes / installedPhysicalBytes * 100,
        usablePerDeviceBytes: usablePerDevice,
    }
}
export function roundAppSizing(value: number, step: number) {
    return Math.ceil(value / step) * step
}
