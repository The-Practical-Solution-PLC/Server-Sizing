import { parseAttentionProfile, validateModelMetadata, type ModelMetadata } from "./calculators.ts"

const HUGGING_FACE_ORIGIN = "https://huggingface.co"
const FETCH_TIMEOUT_MS = 10_000
const MAX_JSON_BYTES = 1_048_576
const REPO_PART = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/
const REVISION = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/

type HuggingFaceModelInfo = {
    id?: string
    author?: string
    sha?: string
    safetensors?: {
        total?: number
    }
}

export type HuggingFaceDraft = ModelMetadata & {
    model: string
    modelType: string
    provider: string
    parametersB: number
}

export type HuggingFaceImportResult = {
    repoId: string
    draft: HuggingFaceDraft
    missingFields: string[]
    warnings: string[]
}

function validRepoPart(value: string, name: string) {
    if (!REPO_PART.test(value) || value === "." || value === "..") throw new Error(`Invalid Hugging Face ${name}`)
    return value
}

export function parseHuggingFaceInput(input: string) {
    const value = input.trim()
    if (!value) throw new Error("Hugging Face repository is required")
    if (value.startsWith("http://") || value.startsWith("https://")) {
        let url: URL
        try {
            url = new URL(value)
        } catch {
            throw new Error("Invalid Hugging Face URL")
        }
        if (url.origin !== HUGGING_FACE_ORIGIN || url.username || url.password || url.search || url.hash) {
            throw new Error("Only canonical huggingface.co config URLs are supported")
        }
        const parts = url.pathname.split("/").filter(Boolean)
        if (parts.length !== 5 || parts[2] !== "raw" || parts[4] !== "config.json") {
            throw new Error("Expected https://huggingface.co/{owner}/{repo}/raw/{revision}/config.json")
        }
        const owner = validRepoPart(parts[0], "owner")
        const repo = validRepoPart(parts[1], "repository")
        if (!REVISION.test(parts[3])) throw new Error("Invalid Hugging Face revision")
        return { repoId: `${owner}/${repo}`, requestedRevision: parts[3] }
    }
    const parts = value.split("/")
    if (parts.length !== 2) throw new Error("Use a repository ID such as Qwen/Qwen3.6-35B-A3B")
    const owner = validRepoPart(parts[0], "owner")
    const repo = validRepoPart(parts[1], "repository")
    return { repoId: `${owner}/${repo}`, requestedRevision: null }
}

async function readBoundedJson(response: Response, label: string) {
    if (response.status >= 300 && response.status < 400) throw new Error(`${label} redirected unexpectedly`)
    if (!response.ok) {
        if (response.status === 404) throw new Error(`${label} was not found`)
        if (response.status === 429) throw new Error("Hugging Face rate limit reached; try again later")
        if (response.status === 401 || response.status === 403) throw new Error(`${label} is private or gated`)
        throw new Error(`${label} request failed (${response.status})`)
    }
    const declaredSize = Number(response.headers.get("content-length"))
    if (Number.isFinite(declaredSize) && declaredSize > MAX_JSON_BYTES) throw new Error(`${label} is too large`)
    if (!response.body) throw new Error(`${label} returned an empty response`)
    const reader = response.body.getReader()
    const chunks: Uint8Array[] = []
    let size = 0
    while (true) {
        const { done, value } = await reader.read()
        if (done) break
        size += value.byteLength
        if (size > MAX_JSON_BYTES) {
            await reader.cancel()
            throw new Error(`${label} is too large`)
        }
        chunks.push(value)
    }
    const bytes = new Uint8Array(size)
    let offset = 0
    for (const chunk of chunks) {
        bytes.set(chunk, offset)
        offset += chunk.byteLength
    }
    try {
        return JSON.parse(new TextDecoder().decode(bytes)) as unknown
    } catch {
        throw new Error(`${label} returned invalid JSON`)
    }
}

function configObjects(root: unknown) {
    if (!root || typeof root !== "object" || Array.isArray(root)) throw new Error("Hugging Face config must be an object")
    const objects: Record<string, unknown>[] = []
    const queue = [{ value: root as Record<string, unknown>, depth: 0 }]
    const visited = new Set<object>()
    while (queue.length && objects.length < 100) {
        const current = queue.shift()!
        if (visited.has(current.value)) continue
        visited.add(current.value)
        objects.push(current.value)
        if (current.depth >= 4) continue
        for (const child of Object.values(current.value)) {
            if (child && typeof child === "object" && !Array.isArray(child)) {
                queue.push({ value: child as Record<string, unknown>, depth: current.depth + 1 })
            }
        }
    }
    return objects
}

function topologyConfig(root: unknown) {
    const objects = configObjects(root)
    const textConfig = (root as Record<string, unknown>).text_config
    if (textConfig && typeof textConfig === "object" && !Array.isArray(textConfig)) {
        objects.unshift(textConfig as Record<string, unknown>)
    }
    const config = objects.find((item) => item.num_hidden_layers != null && item.num_attention_heads != null)
    if (!config) throw new Error("No supported text topology found in config.json")
    return config
}

function integer(config: Record<string, unknown>, key: string) {
    const value = Number(config[key])
    return Number.isSafeInteger(value) && value > 0 ? value : null
}

function weightMetadata(root: Record<string, unknown>, config: Record<string, unknown>) {
    const candidate = config.quantization_config ?? root.quantization_config
    const quantization = candidate && typeof candidate === "object" && !Array.isArray(candidate)
        ? candidate as Record<string, unknown>
        : null
    const method = typeof quantization?.quant_method === "string"
        ? quantization.quant_method.trim().toLowerCase()
        : ""
    const explicitBits = Number(quantization?.bits)
    if (["nvfp4", "mxfp4", "fp4"].includes(method)) {
        return { weightFormat: method.toUpperCase(), weightBits: 4 }
    }
    if (method.includes("fp8")) return { weightFormat: method.toUpperCase(), weightBits: 8 }
    if (method && Number.isFinite(explicitBits) && explicitBits > 0 && explicitBits <= 32) {
        return { weightFormat: method.toUpperCase(), weightBits: explicitBits }
    }
    const dtypeValue = config.dtype ?? root.dtype ?? config.torch_dtype ?? root.torch_dtype
    const dtype = typeof dtypeValue === "string" ? dtypeValue.toLowerCase() : ""
    if (dtype === "bfloat16") return { weightFormat: "BF16", weightBits: 16 }
    if (["float16", "half"].includes(dtype)) return { weightFormat: "FP16", weightBits: 16 }
    return { weightFormat: null, weightBits: null }
}

function attentionMetadata(config: Record<string, unknown>) {
    const q = integer(config, "num_attention_heads")
    const kv = integer(config, "num_key_value_heads") ?? q
    const layers = integer(config, "num_hidden_layers")
    const layerTypes = Array.isArray(config.layer_types)
        ? config.layer_types.filter((item): item is string => typeof item === "string")
        : []
    if (!q || !kv || !layers || kv > q || q % kv !== 0) throw new Error("Invalid attention topology in config.json")
    if (layerTypes.length && layerTypes.length !== layers) throw new Error("layer_types does not match num_hidden_layers")

    const fullLayers = layerTypes.filter((item) => item === "full_attention").length
    const slidingLayers = layerTypes.filter((item) => item === "sliding_attention").length
    const linearLayers = layerTypes.filter((item) => item === "linear_attention").length
    if (fullLayers && slidingLayers && fullLayers + slidingLayers === layers) {
        const slidingWindow = integer(config, "sliding_window")
        if (!slidingWindow) throw new Error("Sliding attention config is missing sliding_window")
        return {
            architecture: "sliding_window",
            attentionProfile: JSON.stringify({ kind: "sliding_full", fullLayers, slidingLayers, slidingWindow }),
        }
    }
    if (fullLayers && linearLayers && fullLayers + linearLayers === layers) {
        const fields = {
            convKernelDim: integer(config, "linear_conv_kernel_dim"),
            keyHeadDim: integer(config, "linear_key_head_dim"),
            valueHeadDim: integer(config, "linear_value_head_dim"),
            numKeyHeads: integer(config, "linear_num_key_heads"),
            numValueHeads: integer(config, "linear_num_value_heads"),
        }
        if (Object.values(fields).some((value) => value === null)) throw new Error("Linear attention config is missing state dimensions")
        return {
            architecture: "hybrid",
            attentionProfile: JSON.stringify({
                kind: "linear_full",
                fullLayers,
                linearLayers,
                linear: { ...fields, convStateBits: 16, recurrentStateBits: 32 },
            }),
        }
    }
    if (layerTypes.length && fullLayers !== layers) throw new Error("Unsupported layer_types attention profile")
    return {
        architecture: q === kv ? "decoder_mha" : kv === 1 ? "decoder_mqa" : "decoder_gqa",
        attentionProfile: null,
    }
}

export function mapHuggingFaceModel(
    repoId: string,
    info: HuggingFaceModelInfo,
    root: unknown,
    revision: string,
    verifiedAt = new Date().toISOString().slice(0, 10),
): HuggingFaceImportResult {
    const rootConfig = root as Record<string, unknown>
    const config = topologyConfig(root)
    const totalParameters = Number(info.safetensors?.total)
    if (!Number.isSafeInteger(totalParameters) || totalParameters <= 0) throw new Error("Hugging Face did not provide a safetensors parameter total")
    const numHiddenLayers = integer(config, "num_hidden_layers")
    const numAttentionHeads = integer(config, "num_attention_heads")
    const numKeyValueHeads = integer(config, "num_key_value_heads") ?? numAttentionHeads
    const headDim = integer(config, "head_dim")
    const hiddenSize = integer(config, "hidden_size")
    const maxContextTokens = integer(config, "max_position_embeddings")
    const attention = attentionMetadata(config)
    const weight = weightMetadata(rootConfig, config)
    const [namespace, slug] = repoId.split("/")
    const metadataSource = `${HUGGING_FACE_ORIGIN}/${repoId}/raw/${revision}/config.json`
    const draft: HuggingFaceDraft = {
        model: slug,
        modelType: "llm",
        provider: info.author?.trim() || namespace,
        parametersB: totalParameters / 1_000_000_000,
        modelId: repoId,
        revision,
        totalParameters,
        architecture: attention.architecture,
        attentionProfile: attention.attentionProfile,
        numHiddenLayers,
        numAttentionHeads,
        numKeyValueHeads,
        headDim,
        hiddenSize,
        maxContextTokens,
        ...(weight.weightFormat && weight.weightBits ? {
            weightFormat: weight.weightFormat,
            weightBits: weight.weightBits,
        } : {}),
        metadataSource,
        verifiedAt,
    }
    validateModelMetadata(draft)
    if (draft.attentionProfile) parseAttentionProfile(draft.attentionProfile)
    const missingFields = ["activeParameters", "weightFormat", "weightBits", "kvCacheBits"].filter((key) => draft[key as keyof HuggingFaceDraft] == null)
    const warnings = ["Active parameters are not published by the selected endpoints and were not inferred from the model name."]
    if (weight.weightFormat && (rootConfig.quantization_config || config.quantization_config)) {
        warnings.push(`${weight.weightFormat} was imported as a ${weight.weightBits}-bit weight estimate; scales, packing and runtime overhead still apply.`)
    } else if (!weight.weightFormat) {
        warnings.push("No supported weight precision metadata was found; sizing will use the FP16 fallback unless overridden.")
    }
    return { repoId, draft, missingFields, warnings }
}

export async function fetchHuggingFaceModelDraft(
    input: string,
    fetcher: typeof fetch = fetch,
    verifiedAt?: string,
) {
    const { repoId, requestedRevision } = parseHuggingFaceInput(input)
    const revisionPath = requestedRevision ? `/revision/${requestedRevision}` : ""
    const apiUrl = new URL(`${HUGGING_FACE_ORIGIN}/api/models/${repoId}${revisionPath}`)
    for (const field of ["author", "sha", "safetensors"]) apiUrl.searchParams.append("expand[]", field)
    const options: RequestInit = { redirect: "manual", signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) }
    let info: HuggingFaceModelInfo
    try {
        info = await readBoundedJson(await fetcher(apiUrl, options), "Hugging Face model") as HuggingFaceModelInfo
    } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") throw new Error("Hugging Face request timed out")
        throw error
    }
    const revision = info.sha
    if (!revision || !REVISION.test(revision)) throw new Error("Hugging Face did not provide a valid revision")
    const configUrl = `${HUGGING_FACE_ORIGIN}/${repoId}/raw/${revision}/config.json`
    let config: unknown
    try {
        config = await readBoundedJson(await fetcher(configUrl, options), "Hugging Face config")
    } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") throw new Error("Hugging Face request timed out")
        throw error
    }
    return mapHuggingFaceModel(repoId, info, config, revision, verifiedAt)
}
