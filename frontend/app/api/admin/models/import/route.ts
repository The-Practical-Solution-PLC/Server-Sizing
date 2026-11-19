import { NextResponse } from "next/server"
import { fetchHuggingFaceModelDraft } from "@/app/lib/huggingface-import"
import { prisma } from "@/app/lib/prisma"

export async function POST(request: Request) {
    try {
        if (!request.headers.get("content-type")?.toLowerCase().includes("application/json")) {
            return NextResponse.json({ message: "Expected application/json" }, { status: 415 })
        }
        const declaredSize = Number(request.headers.get("content-length"))
        if (Number.isFinite(declaredSize) && declaredSize > 10_000) {
            return NextResponse.json({ message: "Request body is too large" }, { status: 413 })
        }
        const reader = request.body?.getReader()
        const decoder = new TextDecoder()
        let text = ""
        let bytesRead = 0
        if (reader) {
            while (true) {
                const { done, value } = await reader.read()
                if (done) break
                bytesRead += value.byteLength
                if (bytesRead > 10_000) {
                    await reader.cancel()
                    return NextResponse.json({ message: "Request body is too large" }, { status: 413 })
                }
                text += decoder.decode(value, { stream: true })
            }
            text += decoder.decode()
        }
        const body = JSON.parse(text) as Record<string, unknown>
        if (typeof body.repoId !== "string" || Object.keys(body).some((key) => key !== "repoId")) {
            return NextResponse.json({ message: "Expected only a repoId string" }, { status: 400 })
        }
        const result = await fetchHuggingFaceModelDraft(body.repoId)
        const existingMatches = await prisma.llmModel.findMany({
            where: { modelId: result.repoId },
            select: { id: true, model: true, isActive: true },
            orderBy: { id: "asc" },
        })
        return NextResponse.json({ ...result, existingMatches })
    } catch (error) {
        return NextResponse.json(
            { message: error instanceof Error ? error.message : "Hugging Face import failed" },
            { status: 400 },
        )
    }
}
