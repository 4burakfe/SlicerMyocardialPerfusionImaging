"""Small Qt helpers shared by the MPI modules."""

# (normal, hover, pressed)
BUTTON_COLORS = {
    "primary": ("#2f7ed8", "#4a92e3", "#2466b3"),   # start / go to the next step
    "action": ("#e0782a", "#ea8f48", "#c2631b"),    # interactive placement
    "create": ("#2e9e5b", "#3cb46c", "#24804a"),    # produces data
}


def styleButton(button, kind="primary"):
    """Coloured main action button (white bold text, grey when disabled)."""
    normal, hover, pressed = BUTTON_COLORS[kind]
    button.setStyleSheet(
        f"QPushButton {{ background-color: {normal}; color: white; font-weight: bold; border: none;"
        f" border-radius: 4px; padding: 6px 12px; }}"
        f"QPushButton:hover {{ background-color: {hover}; }}"
        f"QPushButton:pressed {{ background-color: {pressed}; }}"
        "QPushButton:disabled { background-color: #9aa1a7; color: #eceff1; }")
    button.setMinimumHeight(30)
    return button
