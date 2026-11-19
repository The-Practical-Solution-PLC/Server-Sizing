import { PrismaClient } from "@prisma/client"
import { fetchHuggingFaceModelDraft } from "../app/lib/huggingface-import.ts"

const prisma = new PrismaClient()
const requestedModelIds = process.argv.slice(2)
const models = await prisma.llmModel.findMany({
    where: {
        isActive: true,
        modelType: "llm",
        modelId: requestedModelIds.length ? { in: requestedModelIds } : { not: null },
        revision: { not: null },
    },
    select: {
        id: true,
        model: true,
        modelId: true,
        revision: true,
    },
})
let updated = 0
const failures = []
for (const model of models) {
    try {
        const configUrl = `https://huggingface.co/${model.modelId}/raw/${model.revision}/config.json`
        const { draft } = await fetchHuggingFaceModelDraft(configUrl)
        await prisma.llmModel.update({
            where: { id: model.id },
            data: {
                parametersB: draft.parametersB,
                totalParameters: draft.totalParameters,
                architecture: draft.architecture,
                attentionProfile: draft.attentionProfile,
                numHiddenLayers: draft.numHiddenLayers,
                numAttentionHeads: draft.numAttentionHeads,
                numKeyValueHeads: draft.numKeyValueHeads,
                headDim: draft.headDim,
                hiddenSize: draft.hiddenSize,
                maxContextTokens: draft.maxContextTokens,
                ...(draft.weightFormat && draft.weightBits ? {
                    weightFormat: draft.weightFormat,
                    weightBits: draft.weightBits,
                } : {}),
                metadataSource: draft.metadataSource,
                verifiedAt: draft.verifiedAt,
            },
        })
        updated += 1
    } catch (error) {
        failures.push({ model: model.model, message: error instanceof Error ? error.message : "Refresh failed" })
    }
}
console.log(JSON.stringify({ models: models.length, updated, failures }, null, 2))
await prisma.$disconnect()
