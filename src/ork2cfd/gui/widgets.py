"""Reusable CustomTkinter form widgets with (i) info tooltips and click popups."""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

from .help_texts import HELP_TEXTS, HelpEntry


class InfoIcon(ctk.CTkButton):
    """Compact circular (i) badge showing a hover tooltip and click-to-pin detail popup."""

    def __init__(self, master, help_key: str, **kwargs) -> None:
        self.help_key = help_key
        self._entry: HelpEntry = HELP_TEXTS[help_key]
        self._hover_after_id: str | None = None
        self._tooltip_win: tk.Toplevel | None = None
        self._dialog_win: ctk.CTkToplevel | None = None

        super().__init__(
            master,
            text="i",
            width=18,
            height=18,
            corner_radius=9,
            border_width=1,
            border_color=("#94a3b8", "#475569"),
            fg_color=("#e2e8f0", "#1e293b"),
            hover_color=("#cbd5e1", "#334155"),
            text_color=("#0284c7", "#38bdf8"),
            font=ctk.CTkFont(family="Georgia", size=11, weight="bold", slant="italic"),
            command=self.open_dialog,
            **kwargs,
        )
        with contextlib.suppress(tk.TclError):
            self.configure(cursor="hand2")

        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_enter(self, _event=None) -> None:
        self._cancel_hover_timer()
        self._hover_after_id = self.after(220, self.show_tooltip)

    def _on_leave(self, _event=None) -> None:
        self._cancel_hover_timer()
        self.hide_tooltip()

    def _on_destroy(self, _event=None) -> None:
        self._cancel_hover_timer()
        self.hide_tooltip()

    def _cancel_hover_timer(self) -> None:
        if self._hover_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._hover_after_id)
            self._hover_after_id = None

    def show_tooltip(self) -> None:
        """Display floating hover tooltip clamped inside screen boundaries."""
        self.hide_tooltip()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        bg = "#0f172a" if is_dark else "#ffffff"
        border = "#38bdf8" if is_dark else "#0284c7"
        title_fg = "#38bdf8" if is_dark else "#0369a1"
        body_fg = "#e2e8f0" if is_dark else "#0f172a"
        hint_fg = "#94a3b8" if is_dark else "#64748b"
        div_bg = "#1e293b" if is_dark else "#e2e8f0"

        try:
            win = tk.Toplevel(self)
            win.wm_overrideredirect(True)
            win.wm_attributes("-topmost", True)
        except tk.TclError:
            return

        frame = tk.Frame(
            win,
            bg=bg,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=border,
        )
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text=self._entry.title,
            bg=bg,
            fg=title_fg,
            font=("Segoe UI", 10, "bold"),
            justify="left",
            anchor="w",
            wraplength=360,
        ).pack(fill="x", anchor="w")

        tk.Frame(frame, bg=div_bg, height=1).pack(fill="x", pady=(4, 5))

        tk.Label(
            frame,
            text=self._entry.body,
            bg=bg,
            fg=body_fg,
            font=("Segoe UI", 9),
            justify="left",
            anchor="w",
            wraplength=360,
        ).pack(fill="x", anchor="w")

        tk.Label(
            frame,
            text="Click (i) to pin full guide",
            bg=bg,
            fg=hint_fg,
            font=("Segoe UI", 8, "italic"),
            anchor="e",
        ).pack(fill="x", pady=(5, 0))

        win.update_idletasks()
        w = win.winfo_reqwidth()
        h = win.winfo_reqheight()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()

        x = self.winfo_rootx() + 24
        y = self.winfo_rooty() - 6
        x = max(8, min(x, sw - w - 8))
        y = max(8, min(y, sh - h - 8))
        win.wm_geometry(f"+{x}+{y}")
        self._tooltip_win = win

    def hide_tooltip(self) -> None:
        """Destroy active hover tooltip if open."""
        if self._tooltip_win is not None:
            with contextlib.suppress(tk.TclError):
                self._tooltip_win.destroy()
            self._tooltip_win = None

    def open_dialog(self) -> ctk.CTkToplevel | None:
        """Open (or raise) a pinned scrollable help window for this option."""
        self._cancel_hover_timer()
        self.hide_tooltip()

        if self._dialog_win is not None:
            try:
                if self._dialog_win.winfo_exists():
                    self._dialog_win.lift()
                    self._dialog_win.focus()
                    return self._dialog_win
            except tk.TclError:
                self._dialog_win = None

        try:
            dlg = ctk.CTkToplevel(self.winfo_toplevel())
        except tk.TclError:
            return None

        dlg.title(f"Help — {self._entry.title}")
        dlg.geometry("520x360")
        dlg.minsize(400, 250)
        dlg.grid_columnconfigure(0, weight=1)
        dlg.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            dlg,
            text=self._entry.title,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
            wraplength=480,
        ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="ew")

        textbox = ctk.CTkTextbox(
            dlg,
            wrap="word",
            font=ctk.CTkFont(size=12),
        )
        textbox.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="nsew")
        textbox.insert("1.0", self._entry.body)
        textbox.configure(state="disabled")

        ctk.CTkButton(dlg, text="Close", width=90, command=dlg.destroy).grid(
            row=2, column=0, padx=16, pady=(0, 14), sticky="e"
        )
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        with contextlib.suppress(tk.TclError):
            dlg.after(30, dlg.lift)
        self._dialog_win = dlg
        return dlg


class SectionCard(ctk.CTkFrame):
    """Styled card container with a bold section title and optional (i) info badge."""

    def __init__(
        self,
        master,
        title: str,
        help_key: str | None = None,
        icon_registry: list[InfoIcon] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, corner_radius=8, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            hdr,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        if help_key is not None:
            icon = InfoIcon(hdr, help_key=help_key)
            icon.grid(row=0, column=1, padx=(6, 0), sticky="e")
            if icon_registry is not None:
                icon_registry.append(icon)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.body.grid_columnconfigure((0, 1), weight=1)


def _create_field_header(
    parent,
    label: str,
    help_key: str | None,
    icon_registry: list[InfoIcon] | None,
) -> ctk.CTkFrame:
    """Create label + optional (i) badge row above an input widget."""
    hdr = ctk.CTkFrame(parent, fg_color="transparent")
    hdr.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    ctk.CTkLabel(hdr, text=label, font=ctk.CTkFont(size=11), anchor="w").pack(side="left")
    if help_key is not None:
        icon = InfoIcon(hdr, help_key=help_key)
        icon.pack(side="left", padx=(5, 0))
        if icon_registry is not None:
            icon_registry.append(icon)
    return hdr


def add_labeled_entry(
    parent,
    row: int,
    col: int,
    label: str,
    variable: ctk.StringVar,
    on_change: Callable | None = None,
    help_key: str | None = None,
    icon_registry: list[InfoIcon] | None = None,
) -> ctk.CTkEntry:
    """Create a compact labeled text entry with optional (i) info badge."""
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=col, padx=4, pady=3, sticky="ew")
    frame.grid_columnconfigure(0, weight=1)
    _create_field_header(frame, label, help_key, icon_registry)
    entry = ctk.CTkEntry(frame, textvariable=variable, height=28)
    entry.grid(row=1, column=0, sticky="ew")
    if on_change is not None:
        entry.bind("<FocusOut>", lambda _e: on_change())
        entry.bind("<Return>", lambda _e: on_change())
    return entry


def add_labeled_combo(
    parent,
    row: int,
    col: int,
    label: str,
    values: list[str],
    variable: ctk.StringVar,
    on_change: Callable | None = None,
    help_key: str | None = None,
    icon_registry: list[InfoIcon] | None = None,
) -> ctk.CTkOptionMenu:
    """Create a compact labeled dropdown menu with optional (i) info badge."""
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=col, padx=4, pady=3, sticky="ew")
    frame.grid_columnconfigure(0, weight=1)
    _create_field_header(frame, label, help_key, icon_registry)
    menu = ctk.CTkOptionMenu(
        frame,
        values=values,
        variable=variable,
        height=28,
        command=(lambda _v: on_change()) if on_change else None,
    )
    menu.grid(row=1, column=0, sticky="ew")
    return menu


def add_labeled_checkbox(
    parent,
    row: int,
    col: int,
    text: str,
    variable: ctk.BooleanVar,
    help_key: str | None = None,
    command: Callable | None = None,
    icon_registry: list[InfoIcon] | None = None,
) -> ctk.CTkCheckBox:
    """Create a checkbox paired cleanly with an inline (i) info badge."""
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=col, padx=4, pady=3, sticky="ew")
    cb = ctk.CTkCheckBox(frame, text=text, variable=variable, command=command)
    cb.pack(side="left")
    if help_key is not None:
        icon = InfoIcon(frame, help_key=help_key)
        icon.pack(side="left", padx=(6, 0))
        if icon_registry is not None:
            icon_registry.append(icon)
    return cb
