'use client'

import { ChevronDown } from "lucide-react"
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react"

export type SelectOption = {
    value: string
    label: string
    icon?: ReactNode
}

export default function CustomSelect({
    value,
    options,
    placeholder = "Select",
    ariaLabel = placeholder,
    searchable = false,
    onChange,
}: {
    value: string
    options: SelectOption[]
    placeholder?: string
    ariaLabel?: string
    searchable?: boolean
    onChange: (value: string) => void
}) {
    const [open, setOpen] = useState(false)
    const [search, setSearch] = useState("")
    const ref = useRef<HTMLDivElement | null>(null)
    const buttonRef = useRef<HTMLButtonElement | null>(null)
    const searchRef = useRef<HTMLInputElement | null>(null)
    const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
    const [activeIndex, setActiveIndex] = useState(0)
    const menuId = useId()
    const selected = options.find((option) => option.value === value)
    const filtered = useMemo(() => {
        const query = search.trim().toLowerCase()
        if (!query) return options
        return options.filter((option) => option.label.toLowerCase().includes(query))
    }, [options, search])

    useEffect(() => {
        if (!open) return
        function onPointerDown(event: PointerEvent) {
            if (!ref.current?.contains(event.target as Node)) {
                setOpen(false)
                setSearch("")
            }
        }
        function onKeyDown(event: KeyboardEvent) {
            if (event.key !== "Escape") return
            setOpen(false)
            setSearch("")
            buttonRef.current?.focus()
        }
        document.addEventListener("pointerdown", onPointerDown)
        document.addEventListener("keydown", onKeyDown)
        return () => {
            document.removeEventListener("pointerdown", onPointerDown)
            document.removeEventListener("keydown", onKeyDown)
        }
    }, [open])

    function openMenu() {
        const selectedIndex = filtered.findIndex((option) => option.value === value)
        const nextIndex = selectedIndex < 0 ? 0 : selectedIndex
        setActiveIndex(nextIndex)
        setOpen(true)
        requestAnimationFrame(() => {
            if (searchable) searchRef.current?.focus()
            else optionRefs.current[nextIndex]?.focus()
        })
    }

    function focusOption(index: number) {
        if (!filtered.length) return
        const nextIndex = (index + filtered.length) % filtered.length
        setActiveIndex(nextIndex)
        optionRefs.current[nextIndex]?.focus()
    }

    function onMenuKeyDown(event: ReactKeyboardEvent) {
        if (event.key === "ArrowDown") {
            event.preventDefault()
            focusOption(activeIndex + 1)
        } else if (event.key === "ArrowUp") {
            event.preventDefault()
            focusOption(activeIndex - 1)
        } else if (event.key === "Home") {
            event.preventDefault()
            focusOption(0)
        } else if (event.key === "End") {
            event.preventDefault()
            focusOption(filtered.length - 1)
        }
    }

    return (
        <div className="custom-select" ref={ref}>
            <button
                type="button"
                ref={buttonRef}
                className="custom-select-button"
                aria-label={ariaLabel}
                aria-expanded={open}
                aria-controls={menuId}
                aria-haspopup="listbox"
                onClick={() => {
                    if (open) {
                        setOpen(false)
                        setSearch("")
                    } else openMenu()
                }}
                onKeyDown={(event) => {
                    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
                    event.preventDefault()
                    openMenu()
                }}
            >
                <span className="custom-select-value">{selected?.icon}{selected?.label ?? placeholder}</span>
                <ChevronDown size={18} />
            </button>
            {open ? (
                <div className="custom-select-menu" onKeyDown={onMenuKeyDown}>
                    {searchable ? (
                        <input
                            className="input"
                            ref={searchRef}
                            aria-label={`${ariaLabel} search`}
                            value={search}
                            placeholder="Search"
                            onChange={(event) => {
                                setSearch(event.target.value)
                                setActiveIndex(0)
                            }}
                        />
                    ) : null}
                    <div id={menuId} role="listbox">
                        {filtered.map((option, index) => (
                            <button
                                type="button"
                                role="option"
                                aria-selected={option.value === value}
                                ref={(node) => { optionRefs.current[index] = node }}
                                tabIndex={index === activeIndex ? 0 : -1}
                                key={option.value}
                                className={`custom-select-option ${option.value === value ? "active" : ""}`}
                                onClick={() => {
                                    onChange(option.value)
                                    setOpen(false)
                                    setSearch("")
                                    buttonRef.current?.focus()
                                }}
                            >
                                <span className="custom-select-value">{option.icon}{option.label}</span>
                            </button>
                        ))}
                        {filtered.length === 0 ? <div className="custom-select-option muted">No data</div> : null}
                    </div>
                </div>
            ) : null}
        </div>
    )
}
