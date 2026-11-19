'use client'

import { X } from "lucide-react"
import { useEffect, useEffectEvent, useRef } from "react"

export default function Drawer({
    title,
    actions,
    open,
    children,
    onClose,
}: {
    title: string
    actions?: React.ReactNode
    open: boolean
    children: React.ReactNode
    onClose: () => void
}) {
    const closeButtonRef = useRef<HTMLButtonElement | null>(null)
    const drawerRef = useRef<HTMLElement | null>(null)
    const previousFocusRef = useRef<HTMLElement | null>(null)
    const close = useEffectEvent(onClose)

    useEffect(() => {
        if (!open) return
        previousFocusRef.current = document.activeElement as HTMLElement | null
        closeButtonRef.current?.focus()
        function onKeyDown(event: KeyboardEvent) {
            if (event.key === "Escape") {
                if (drawerRef.current?.querySelector('[aria-expanded="true"]')) return
                close()
            }
            if (event.key !== "Tab") return
            const focusable = drawerRef.current?.querySelectorAll<HTMLElement>(
                'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
            )
            if (!focusable?.length) return
            const first = focusable[0]
            const last = focusable[focusable.length - 1]
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault()
                last.focus()
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault()
                first.focus()
            }
        }
        document.addEventListener("keydown", onKeyDown)
        return () => {
            document.removeEventListener("keydown", onKeyDown)
            previousFocusRef.current?.focus()
        }
    }, [open])

    if (!open) return null

    return (
        <>
            <button
                type="button"
                tabIndex={-1}
                aria-label="Close drawer"
                className="drawer-backdrop"
                onClick={onClose}
            />
            <aside ref={drawerRef} className="drawer" role="dialog" aria-modal="true" aria-label={title}>
                <div className="drawer-header">
                    <h2>{title}</h2>
                    <button ref={closeButtonRef} type="button" className="drawer-close-button" onClick={onClose} aria-label="Close">
                        <X size={20} />
                    </button>
                </div>
                <div className="drawer-body">
                    {children}
                </div>
                {actions ? (
                    <div className="drawer-footer">
                        {actions}
                    </div>
                ) : null}
            </aside>
        </>
    )
}
