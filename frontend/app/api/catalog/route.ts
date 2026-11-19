import { NextResponse } from "next/server"
import { prisma } from "@/app/lib/prisma"

export async function GET() {
    const [models, gpus, apps] = await Promise.all([
        prisma.llmModel.findMany({ where: { isActive: true }, orderBy: [{ provider: "asc" }, { model: "asc" }] }),
        prisma.gpu.findMany({ where: { isActive: true }, orderBy: [{ vendor: "asc" }, { vramGb: "asc" }] }),
        prisma.serverApp.findMany({ where: { isActive: true }, orderBy: { name: "asc" } }),
    ])

    return NextResponse.json({ models, gpus, apps })
}
