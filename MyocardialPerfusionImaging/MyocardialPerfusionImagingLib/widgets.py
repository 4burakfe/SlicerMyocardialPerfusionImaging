"""Small Qt helpers shared by the MPI modules."""

# (normal, hover, pressed)
BUTTON_COLORS = {
    "primary": ("#2f7ed8", "#4a92e3", "#2466b3"),   # start / go to the next step
    "action": ("#e0782a", "#ea8f48", "#c2631b"),    # interactive placement
    "create": ("#2e9e5b", "#3cb46c", "#24804a"),    # produces data
}


ZERO_WIDTH_SPACE = "\u200b"


def breakable(text):
    """Text that may wrap after _ - . / : long node names without spaces would otherwise not wrap."""
    for character in "_-./":
        text = text.replace(character, character + ZERO_WIDTH_SPACE)
    return text


def narrowLabel(label):
    """A wrapped label that takes the width it is given and never widens the module panel (a long node
    name would otherwise make the panel wider than Slicer lets it be, cutting off the controls)."""
    import qt
    label.wordWrap = True
    policy = label.sizePolicy
    policy.setHorizontalPolicy(qt.QSizePolicy.Ignored)
    label.setSizePolicy(policy)
    label.setMinimumWidth(0)
    return label


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


class StudyPairSelector:
    """Study 1 (stress) and Study 2 (rest) of the two-study screens, chosen from the studies of the
    scene (role None left out). Filled from the roles at first; a pick here is shared by all steps
    (session.stressRestStudies). Call refresh() when the step is entered and cleanup() at the end."""

    LABELS = {"stress": "Study 1 (stress):", "rest": "Study 2 (rest):"}

    def __init__(self, parentLayout):
        import qt
        from MyocardialPerfusionImagingLib import session
        self.session = session
        self._filling = False
        self._ids = []
        self.widget = qt.QFrame()
        form = qt.QFormLayout(self.widget)
        form.setContentsMargins(0, 0, 0, 0)
        self.combos = {}
        for slot in session.STUDY_SLOTS:
            combo = qt.QComboBox()
            combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.toolTip = ("Volume shown as the stress study (top / first)" if slot == "stress"
                             else "Volume shown as the rest study (bottom / second)")
            combo.connect("activated(int)", lambda index, s=slot: self.onActivated(s, index))
            self.combos[slot] = combo
            row = qt.QHBoxLayout()
            row.addWidget(combo, 1)
            if slot == "stress":
                swap = qt.QPushButton("⇅")
                swap.toolTip = "Swap Study 1 and Study 2"
                swap.setMaximumWidth(32)
                swap.connect("clicked(bool)", lambda _: self.onSwap())
                row.addWidget(swap)
            else:
                reset = qt.QPushButton("By role")
                reset.toolTip = "Pick Study 1 and Study 2 from the roles set in Data roles again"
                reset.connect("clicked(bool)", lambda _: session.resetStudyPair())
                row.addWidget(reset)
            form.addRow(self.LABELS[slot], row)
        parentLayout.addWidget(self.widget)
        session.addListener(self.refresh)
        self.refresh()

    def refresh(self):
        try:
            candidates = self.session.candidateStudies()
            pair = self.session.stressRestStudies()
        except Exception:
            return
        from MyocardialPerfusionImagingLib import studies as studiesLib
        self._filling = True
        try:
            self._ids = [""] + [n.GetID() for n in candidates]
            for slot, combo in self.combos.items():
                combo.clear()
                combo.addItem("(none)")
                for node in candidates:
                    role, guessed = studiesLib.studyRole(node)
                    combo.addItem(f"{role}{'?' if guessed else ''} · {node.GetName()}")
                current = pair.get(slot)
                combo.setCurrentIndex(self._ids.index(current.GetID()) if current is not None
                                      and current.GetID() in self._ids else 0)
        finally:
            self._filling = False

    def onActivated(self, slot, index):
        if self._filling or not (0 <= index < len(self._ids)):
            return
        import slicer
        node = slicer.mrmlScene.GetNodeByID(self._ids[index]) if self._ids[index] else None
        self.session.setStudySlot(slot, node)

    def onSwap(self):
        pair = self.session.stressRestStudies()
        self.session.setStudyPair(pair["rest"], pair["stress"])

    def cleanup(self):
        self.session.removeListener(self.refresh)
