import AdminCrudPage from "@/app/components/AdminCrudPage"
import { metadataNumbers, metadataStrings } from "@/app/lib/calculators"

export default function AdminModelsPage() {
    return <AdminCrudPage title="Model Data" resource="models" enableHuggingFaceImport fields={[
        { key: "model", label: "Model", required: true },
        { key: "modelType", label: "Type", required: true },
        { key: "provider", label: "Provider", required: true },
        { key: "parametersB", label: "Parameters B", type: "number", required: true },
        { key: "architecture", label: "Architecture", placeholder: "decoder_mha, decoder_gqa, decoder_mqa, mla, sliding_window, hybrid, encoder, multimodal" },
        { key: "maxContextTokens", label: "Max Context", type: "number" },
        { key: "activeParameters", label: "Active Parameters", type: "number" },
        { key: "weightFormat", label: "Weight Format" },
        { key: "weightBits", label: "Weight Bits", type: "number" },
        { key: "kvCacheBits", label: "KV Cache Bits", type: "number" },
        { key: "verifiedAt", label: "Verified At" },
        ...metadataNumbers.filter((key) => !["maxContextTokens", "activeParameters", "weightBits", "kvCacheBits"].includes(key)).map((key) => ({ key, label: `${key} (nullable)`, type: "number" as const, hiddenInTable: true })),
        ...metadataStrings.filter((key) => !["architecture", "weightFormat", "verifiedAt"].includes(key)).map((key) => ({
            key,
            label: `${key} (nullable)`,
            type: key === "attentionProfile" ? "textarea" as const : undefined,
            hiddenInTable: true,
        })),
    ]} />
}
