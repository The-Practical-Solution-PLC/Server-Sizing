'use client'

import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react"
import {
    getCoreRowModel,
    getFilteredRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    useReactTable,
    type ColumnDef,
    type ColumnFiltersState,
    type RowSelectionState,
    type SortingState,
} from "@tanstack/react-table"
import CustomSelect from "@/app/components/CustomSelect"
import Drawer from "@/app/components/Drawer"
import InlineLoader from "@/app/components/InlineLoader"
import Loading from "@/app/components/Loading"
import ProviderLogo, { AppNameWithLogo, NameWithLogo } from "@/app/components/ProviderLogo"
import Table from "@/app/components/Table/Table"
import "@/app/style/table.css"

export type AdminField = {
    key: string
    label: string
    type?: "text" | "number" | "url" | "textarea" | "boolean"
    required?: boolean
    placeholder?: string
    hiddenInTable?: boolean
}

type Row = Record<string, string | number | boolean | null>
type Mode = "view" | "edit" | null
type HuggingFacePreview = {
    repoId: string
    draft: Row
    missingFields: string[]
    warnings: string[]
    existingMatches: { id: number, model: string, isActive: boolean }[]
}

const metricKeys = new Set(["parametersB", "vramGb", "networkGbps", "cpuCores", "ramGb", "storageGb"])
const metaKeys = new Set(["provider", "vendor", "modelType", "category", "className", "isActive"])
const titlePriority = ["name", "model", "modelName", "gpuModel", "profileName", "label"]

export default function AdminCrudPage({
    title,
    resource,
    fields,
    enableHuggingFaceImport = false,
}: {
    title: string
    resource: string
    fields: AdminField[]
    enableHuggingFaceImport?: boolean
}) {
    const [rows, setRows] = useState<Row[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [saveError, setSaveError] = useState("")
    const [deleting, setDeleting] = useState(false)
    const [mode, setMode] = useState<Mode>(null)
    const [editing, setEditing] = useState<Row | null>(null)
    const [form, setForm] = useState<Row>({})
    const [sorting, setSorting] = useState<SortingState>([])
    const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
    const [globalFilter, setGlobalFilter] = useState("")
    const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
    const [repoId, setRepoId] = useState("")
    const [importing, setImporting] = useState(false)
    const [importError, setImportError] = useState("")
    const [importPreview, setImportPreview] = useState<HuggingFacePreview | null>(null)

    async function load() {
        setLoading(true)
        const data = await fetch(`/api/admin/${resource}`).then((response) => response.json())
        setRows(data)
        setLoading(false)
    }

    useEffect(() => {
        load()
        setRowSelection({})
        setMode(null)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [resource])

    const visibleFields = useMemo(() => fields.filter((field) => field.key !== "isActive" && !field.hiddenInTable), [fields])
    const selectOptions = useMemo(() => {
        const providerValues = distinct(rows.map((row) => row.provider))
        const vendorValues = distinct(rows.map((row) => row.vendor))
        return { provider: providerValues, vendor: vendorValues }
    }, [rows])

    function openAdd() {
        setEditing(null)
        setForm(Object.fromEntries(fields.map((field) => [field.key, field.type === "boolean" ? true : ""])))
        setMode("edit")
        resetImport()
    }

    function openView(row: Row) {
        setEditing(row)
        setForm(row)
        setMode("view")
    }

    function openEdit(row: Row) {
        setEditing(row)
        setForm(row)
        setMode("edit")
        setRepoId(String(row.modelId ?? ""))
        setImportError("")
        setImportPreview(null)
    }

    function closeDrawer() {
        setMode(null)
        setEditing(null)
        setForm({})
        resetImport()
    }

    function resetImport() {
        setRepoId("")
        setImportError("")
        setImportPreview(null)
    }

    async function fetchHuggingFaceMetadata() {
        if (importing || !repoId.trim()) return
        setImporting(true)
        setImportError("")
        setImportPreview(null)
        try {
            const response = await fetch("/api/admin/models/import", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repoId: repoId.trim() }),
            })
            const data = await response.json()
            if (!response.ok) throw new Error(data.message ?? "Hugging Face import failed")
            setImportPreview(data)
        } catch (error) {
            setImportError(error instanceof Error ? error.message : "Hugging Face import failed")
        } finally {
            setImporting(false)
        }
    }

    function applyHuggingFaceMetadata() {
        if (!importPreview) return
        setForm((current) => ({
            ...current,
            ...importPreview.draft,
            model: current.model || importPreview.draft.model,
            provider: current.provider || importPreview.draft.provider,
        }))
        setRepoId(importPreview.repoId)
        setImportPreview(null)
    }

    function payloadFromForm() {
        const payload = { ...form }
        for (const field of fields) {
            const value = payload[field.key]
            if (field.type === "number") payload[field.key] = value === "" || value === null ? (field.required ? 0 : null) : Number(value)
            if (field.type === "boolean") payload[field.key] = Boolean(value)
            if ((field.type === "text" || field.type === "url" || field.type === "textarea" || !field.type) && value === "") payload[field.key] = field.required ? "" : null
        }
        return payload
    }

    async function save() {
        if (saving) return
        setSaving(true)
        setSaveError("")
        try {
            const payload = payloadFromForm()
            const data = Object.fromEntries(fields.map((field) => [field.key, payload[field.key]]))
            for (const field of fields) {
                if (field.required && (data[field.key] === "" || data[field.key] == null)) throw new Error(`${field.label} is required`)
                if (typeof data[field.key] === "number" && !Number.isFinite(data[field.key])) throw new Error(`${field.label} must be finite`)
            }
            const response = await fetch(editing ? `/api/admin/${resource}/${editing.id}` : `/api/admin/${resource}`, {
                method: editing ? "PATCH" : "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            })
            if (!response.ok) throw new Error((await response.json()).message ?? "Save failed")
            closeDrawer()
            await load()
        } catch (error) {
            setSaveError(error instanceof Error ? error.message : "Save failed")
        } finally {
            setSaving(false)
        }
    }

    async function softDelete(row: Row) {
        if (deleting || !window.confirm(`Delete ${String(row.name ?? row.model ?? row.id)}?`)) return
        setDeleting(true)
        await fetch(`/api/admin/${resource}/${row.id}`, { method: "DELETE" })
        setDeleting(false)
        closeDrawer()
        await load()
    }

    async function bulkDelete() {
        const selected = table.getSelectedRowModel().rows.map((row) => row.original)
        if (selected.length === 0 || !window.confirm(`Delete ${selected.length} selected record(s)?`)) return
        setDeleting(true)
        await Promise.all(selected.map((row) => fetch(`/api/admin/${resource}/${row.id}`, { method: "DELETE" })))
        setDeleting(false)
        setRowSelection({})
        await load()
    }

    function renderCell(row: Row, field: AdminField) {
        const value = row[field.key]
        if (field.key === "provider" || field.key === "vendor") return <NameWithLogo name={String(value ?? "")} showText={true} />
        if (resource === "apps" && field.key === "name") return <AppNameWithLogo name={String(row.name ?? "")} iconName={String(row.iconName ?? "")} iconBackground={String(row.iconBackground ?? "auto")} />
        if (field.key === "modelType" || field.key === "category" || field.key === "className") {
            const strVal = String(value ?? "").toUpperCase()
            if (!strVal || strVal === "-") return "-"
            const chipColor = strVal === "OCR" ? "table-chip-yellow" : strVal === "EMBEDDING" ? "table-chip-blue" : "table-chip-gray"
            return <span className={`table-chip ${chipColor}`}>{strVal}</span>
        }
        if (field.type === "url") {
            if (!value) return "-"
            return (
                <a
                    href={String(value)}
                    target="_blank"
                    rel="noreferrer"
                    className="table-chip table-chip-gray hover:text-blue-600 dark:hover:text-blue-400 transition inline-flex items-center gap-1"
                    title={String(value)}
                    onClick={(e) => e.stopPropagation()}
                >
                    <span>Link</span>
                    <span className="text-[10px]">↗</span>
                </a>
            )
        }

        const isNumeric = typeof value === "number" || field.type === "number" || metricKeys.has(field.key)
        if (value === null || value === undefined || value === "") return <span className="text-slate-400 dark:text-slate-600">-</span>

        if (isNumeric) {
            if (field.key === "parametersB" || field.key === "maxModelB" || field.key === "modelBucketB") {
                return <span className="font-mono font-medium">{formatValue(value, field)}</span>
            }
            return <span className="font-mono">{formatValue(value, field)}</span>
        }

        return formatValue(value, field)
    }

    const columns: ColumnDef<Row>[] = [
        {
            id: "select",
            header: ({ table }) => (
                <input type="checkbox" className="table-checkbox" checked={table.getIsAllPageRowsSelected()} onChange={table.getToggleAllPageRowsSelectedHandler()} onClick={(event) => event.stopPropagation()} aria-label="Select all" />
            ),
            cell: ({ row }) => (
                <input type="checkbox" className="table-checkbox" checked={row.getIsSelected()} onChange={row.getToggleSelectedHandler()} onClick={(event) => event.stopPropagation()} aria-label="Select row" />
            ),
            enableSorting: false,
            meta: { width: 48, align: "center" },
        },
        ...visibleFields.map<ColumnDef<Row>>((field) => ({
            accessorKey: field.key,
            header: field.label,
            cell: ({ row }) => renderCell(row.original, field),
        })),
    ]

    const table = useReactTable({
        data: rows,
        columns,
        initialState: {
            pagination: { pageSize: 25 },
        },
        state: { sorting, columnFilters, globalFilter, rowSelection },
        onSortingChange: setSorting,
        onColumnFiltersChange: setColumnFilters,
        onGlobalFilterChange: setGlobalFilter,
        onRowSelectionChange: setRowSelection,
        getRowId: (row) => String(row.id),
        getCoreRowModel: getCoreRowModel(),
        getFilteredRowModel: getFilteredRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
    })

    const selectedCount = Object.keys(rowSelection).length
    const drawerTitle = mode === "view" ? titleFor(form) : editing ? "Edit Record" : "Add New Record"
    const drawerActions = mode === "view" ? (
        <>
            <button type="button" className="button secondary" onClick={() => editing && openEdit(editing)} disabled={deleting}>Edit</button>
            {editing ? (
                <button type="button" className="button danger" onClick={() => softDelete(editing)} disabled={deleting}>
                    {deleting ? <InlineLoader /> : "Delete"}
                </button>
            ) : null}
        </>
    ) : mode === "edit" ? (
        <>
            <button type="button" className="button secondary" onClick={closeDrawer} disabled={saving}>Cancel</button>
            <button type="button" id="confirm" disabled={saving} onClick={save}>
                {saving ? <InlineLoader /> : editing ? "Save Changes" : "Create Record"}
            </button>
        </>
    ) : null

    if (loading) return <Loading />

    return (
        <section className="page-stack table-page">
            <div className="page-title-row">
                <h1>{title}</h1>
            </div>
            <Table
                table={table}
                onRowClick={(row) => openView(row.original)}
                header={<>
                    <div className="table-search"><input className="table-search-input" placeholder="Search records..." value={globalFilter} onChange={(event) => setGlobalFilter(event.target.value)} /></div>
                    <div className="table-header-actions">
                        {selectedCount > 0 ? (
                            <button type="button" className="button danger" onClick={bulkDelete} disabled={deleting}>
                                {deleting ? <InlineLoader /> : `Delete selected (${selectedCount})`}
                            </button>
                        ) : null}
                        <button type="button" id="confirm" onClick={openAdd}>Add New</button>
                    </div>
                </>}
            />
            <Drawer title={drawerTitle} actions={drawerActions} open={mode !== null} onClose={closeDrawer}>
                {saveError ? <p role="alert">{saveError}</p> : null}
                {mode === "view" ? <ViewDetails fields={fields} row={form} resource={resource} /> : (
                    <div className="page-stack">
                        {enableHuggingFaceImport ? (
                            <HuggingFaceImport
                                repoId={repoId}
                                setRepoId={setRepoId}
                                importing={importing}
                                error={importError}
                                preview={importPreview}
                                onFetch={fetchHuggingFaceMetadata}
                                onApply={applyHuggingFaceMetadata}
                            />
                        ) : null}
                        <EditForm fields={fields} form={form} setForm={setForm} selectOptions={selectOptions} />
                    </div>
                )}
            </Drawer>
        </section>
    )
}

function HuggingFaceImport({
    repoId,
    setRepoId,
    importing,
    error,
    preview,
    onFetch,
    onApply,
}: {
    repoId: string
    setRepoId: (value: string) => void
    importing: boolean
    error: string
    preview: HuggingFacePreview | null
    onFetch: () => void
    onApply: () => void
}) {
    return (
        <section className="drawer-section hf-import-section" aria-labelledby="hf-import-title">
            <div className="hf-import-heading">
                <div>
                    <p className="section-kicker">Catalog source</p>
                    <h3 id="hf-import-title">Import from Hugging Face</h3>
                </div>
                <span className="table-chip table-chip-blue">Preview only</span>
            </div>
            <p className="muted">Enter a repository ID. Metadata is fetched from a revision-pinned config before it is applied to this form.</p>
            <form className="hf-import-form" onSubmit={(event) => { event.preventDefault(); onFetch() }}>
                <div className="field">
                    <label htmlFor="hf-repository">Repository or config URL</label>
                    <input
                        id="hf-repository"
                        className="input"
                        value={repoId}
                        placeholder="Qwen/Qwen3.6-35B-A3B"
                        disabled={importing}
                        onChange={(event) => setRepoId(event.target.value)}
                    />
                </div>
                <button type="submit" className="button secondary" disabled={importing || !repoId.trim()}>
                    {importing ? <InlineLoader /> : "Fetch metadata"}
                </button>
            </form>
            {error ? <p role="alert" className="hf-import-message error">{error}</p> : null}
            {preview ? (
                <div className="hf-import-preview" role="status">
                    <div className="hf-import-preview-header">
                        <div>
                            <span className="section-kicker">Pinned preview</span>
                            <strong>{preview.repoId}</strong>
                        </div>
                        <button type="button" id="confirm" onClick={onApply}>Apply metadata</button>
                    </div>
                    <div className="hf-import-metrics">
                        <span><strong>{Number(preview.draft.parametersB).toFixed(2)}B</strong> parameters</span>
                        <span><strong>{String(preview.draft.architecture)}</strong> architecture</span>
                        <span><strong>{preview.draft.maxContextTokens ? Number(preview.draft.maxContextTokens).toLocaleString() : "Unknown"}</strong> context</span>
                        <span><strong>{preview.draft.weightFormat ? `${String(preview.draft.weightFormat)} (${String(preview.draft.weightBits)}-bit)` : "Unknown"}</strong> weight precision</span>
                    </div>
                    {preview.existingMatches.length ? (
                        <p className="hf-import-message warning">{preview.existingMatches.length} catalog record(s) already use this repository. Variants are allowed.</p>
                    ) : null}
                    {preview.warnings.map((warning) => <p className="hf-import-message warning" key={warning}>{warning}</p>)}
                </div>
            ) : null}
        </section>
    )
}

function ViewDetails({ fields, row, resource }: { fields: AdminField[], row: Row, resource: string }) {
    const metrics = fields.filter((field) => field.type === "number" || metricKeys.has(field.key))
    const details = fields.filter((field) => !metrics.includes(field))
    const providerField = fields.find((field) => field.key === "provider" || field.key === "vendor")
    const logoName = providerField ? String(row[providerField.key] ?? "") : resource === "apps" ? String(row.name ?? "") : ""

    return (
        <div className="page-stack">
            <div className="drawer-section drawer-hero">
                <div>
                    <p className="muted">Record Details</p>
                    <h2>{titleFor(row)}</h2>
                </div>
                {logoName ? resource === "apps" ? <AppNameWithLogo name={logoName} iconName={String(row.iconName ?? "")} iconBackground={String(row.iconBackground ?? "auto")} /> : <NameWithLogo name={logoName} /> : null}
            </div>
            {metrics.length ? (
                <div className="drawer-section">
                    <h3>Metrics</h3>
                    <div className="metric-grid">
                        {metrics.map((field) => (
                            <div className="metric-card" key={field.key}>
                                <span>{field.label}</span>
                                <strong className="font-mono">{formatValue(row[field.key], field)}</strong>
                            </div>
                        ))}
                    </div>
                </div>
            ) : null}
            <div className="drawer-section">
                <h3>Specifications</h3>
                <div className="detail-grid">
                    {details.map((field) => (
                        <div className="detail-row" key={field.key}>
                            <span>{field.label}</span>
                            <span>{renderDetailValue(row[field.key], field)}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

function EditForm({ fields, form, setForm, selectOptions }: { fields: AdminField[], form: Row, setForm: Dispatch<SetStateAction<Row>>, selectOptions: { provider: string[], vendor: string[] } }) {
    return (
        <div className="grid-2" style={{ gap: "0.85rem" }}>
            {fields.map((field) => {
                const options = field.key === "provider" ? selectOptions.provider : field.key === "vendor" ? selectOptions.vendor : field.key === "iconBackground" ? ["auto", "light", "dark"] : []
                const isFullWidth = field.type === "textarea" || field.type === "url" || field.key === "note" || field.key === "model" || field.key === "name"
                const placeholder = getFieldPlaceholder(field)
                return (
                    <div className="field" key={field.key} style={isFullWidth ? { gridColumn: "1 / -1" } : undefined}>
                        <label>{field.label}</label>
                        {options.length ? (
                            <CustomSelect value={String(form[field.key] ?? "")} placeholder={placeholder} ariaLabel={field.label} searchable={field.key !== "iconBackground"} options={options.map((value) => ({ value, label: value, icon: field.key === "iconBackground" ? undefined : <ProviderLogo name={value} /> }))} onChange={(value) => setForm((current) => ({ ...current, [field.key]: value }))} />
                        ) : field.type === "boolean" ? (
                            <input type="checkbox" checked={Boolean(form[field.key])} onChange={(event) => setForm((current) => ({ ...current, [field.key]: event.target.checked }))} />
                        ) : field.type === "textarea" ? (
                            <textarea className="textarea" placeholder={placeholder} required={field.required} value={String(form[field.key] ?? "")} onChange={(event) => setForm((current) => ({ ...current, [field.key]: event.target.value }))} />
                        ) : (
                            <input className="input" type={field.type ?? "text"} placeholder={placeholder} required={field.required} min={field.type === "number" ? 0 : undefined} value={String(form[field.key] ?? "")} onChange={(event) => setForm((current) => ({ ...current, [field.key]: event.target.value }))} />
                        )}
                    </div>
                )
            })}
        </div>
    )
}

function getFieldPlaceholder(field: AdminField): string {
    if (field.placeholder) return field.placeholder
    if (field.type === "number") return "0"
    if (field.type === "url") return "https://..."
    if (field.type === "textarea") return `Enter ${field.label.toLowerCase()} details...`
    if (field.key === "model") return "e.g. Llama-3-70B-Instruct"
    if (field.key === "name") return "Enter name..."
    if (field.key === "provider" || field.key === "vendor") return `Select ${field.label.toLowerCase()}...`
    return `Enter ${field.label.toLowerCase()}...`
}

function titleFor(row: Row) {
    for (const key of titlePriority) {
        if (row[key]) return String(row[key])
    }
    return row.id ? `#${row.id}` : "Record"
}

function renderDetailValue(value: Row[string], field: AdminField) {
    if (field.key === "provider" || field.key === "vendor") return <NameWithLogo name={String(value ?? "")} />
    if (metaKeys.has(field.key)) return <span className="table-chip table-chip-gray">{formatValue(value, field)}</span>
    return formatValue(value, field)
}

function formatValue(value: Row[string], field: AdminField) {
    if (value === null || value === undefined || value === "") return "-"
    if (typeof value === "number") {
        if (field.key === "parametersB") return `${value.toFixed(2)}B`
        if (field.key === "totalParameters" || field.key === "activeParameters") return `${(value / 1_000_000_000).toFixed(2)}B`
        const formatted = Number.isInteger(value) ? value : Number(value.toFixed(2))
        if (field.key === "networkGbps") return `${formatted} Gbps`
        if (field.key === "maxModelB" || field.key === "modelBucketB") return `${formatted}B`
        if (field.key.toLowerCase().includes("gb")) return `${formatted}GB`
        return String(formatted)
    }
    return String(value)
}

function distinct(values: Array<Row[string]>) {
    return Array.from(new Set(values.map((value) => String(value ?? "")).filter(Boolean))).sort()
}
