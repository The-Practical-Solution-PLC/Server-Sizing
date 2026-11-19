const fs = require("fs")
const path = require("path")
const { PrismaClient } = require("@prisma/client")
const { parseContext } = require("./context-parser.cjs")

const prisma = new PrismaClient()
const root = path.resolve(__dirname, "..", "..")

function parseNumber(value) {
  if (!value || value === "-") return null
  const normalized = String(value).replace(/,/g, "").trim()
  const parsed = Number(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

function parseMarkdownModels() {
  const file = fs.readFileSync(path.join(root, "local_models.md"), "utf8")
  return file
    .split(/\r?\n/)
    .filter((line) => line.startsWith("|") && !line.includes("---") && !line.includes("rank |"))
    .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim()))
    .filter((cells) => cells.length >= 14)
    .map((cells) => ({
      model: cells[1],
      modelType: "llm",
      provider: cells[2],
      maxContextTokens: parseContext(cells[3]),
      parametersB: parseNumber(cells[4]) || 0,
    }))
}

async function main() {
  const sizing = JSON.parse(fs.readFileSync(path.join(root, "hardware_sizing_data.json"), "utf8"))
  const models = [
    ...parseMarkdownModels().filter((model) => model.parametersB <= 150),
    { model: "Typhoon OCR 2B", modelType: "ocr", provider: "Typhoon", maxContextTokens: null, parametersB: 2 },
    { model: "bge-m3", modelType: "embedding", provider: "BAAI", maxContextTokens: 8192, parametersB: 0.568 },
  ]

  await prisma.$transaction(async (tx) => {
    const existing = await Promise.all([
      tx.llmModel.count(),
      tx.gpu.count(),
      tx.serverApp.count(),
    ])
    if (existing.some((count) => count > 0)) {
      throw new Error("Database is not empty. Seed stopped to avoid overwriting data.")
    }

    await tx.llmModel.createMany({ data: models })
    await tx.gpu.createMany({
      data: sizing.gpuCatalog.map((gpu) => ({
        id: gpu.id,
        vendor: gpu.vendor,
        name: gpu.name,
        className: gpu.class,
        vramGb: gpu.vramGb,
      })),
    })
    await tx.serverApp.createMany({
      data: [
        { name: "Nginx Proxy", category: "Proxy", iconName: "nginx", iconBackground: "auto", networkGbps: 1, cpuCores: 1, ramGb: 1, storageGb: 10, note: "Reverse proxy baseline" },
        { name: "PostgreSQL", category: "Database", iconName: "postgresql", iconBackground: "auto", networkGbps: 1, cpuCores: 4, ramGb: 16, storageGb: 200, note: "General production baseline" },
        { name: "Qdrant", category: "Vector DB", iconName: "qdrant", iconBackground: "dark", networkGbps: 1, cpuCores: 4, ramGb: 16, storageGb: 200, note: "Vector search baseline" },
      ],
    })
  })

  console.log(`Seeded ${models.length} models, ${sizing.gpuCatalog.length} GPUs, and default apps.`)
}

main()
  .finally(async () => prisma.$disconnect())
