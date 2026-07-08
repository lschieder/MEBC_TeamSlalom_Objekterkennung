"""
YOLO Annotator — professionelles Label-Tool
- Box zeichnen: klicken + ziehen
- Klasse wählen: Tab (durchrotieren) oder Zahlentasten / Klick im Klassenpanel
- Box löschen: Rechtsklick auf Box, [D] (ausgewählte/letzte), Doppelklick in Liste
- Speichern + weiter: [S]   Zurück: [A]
- Auto-Label: Modell (.pt) wählen, dann [L]
"""

import tkinter as tk
from tkinter import Canvas, Label, Frame, Listbox, Scrollbar, Button, filedialog, messagebox
from collections import Counter
from PIL import Image, ImageTk, ImageOps
from pathlib import Path

# Catppuccin Mocha
C = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "overlay":  "#6c7086",
    "red":      "#f38ba8",
    "green":    "#a6e3a1",
    "blue":     "#89b4fa",
    "yellow":   "#f9e2af",
    "pink":     "#f5c2e7",
    "lavender": "#b4befe",
    "peach":    "#fab387",
    "teal":     "#94e2d5",
    "mauve":    "#cba6f7",
}

# Boxfarbe je Klassenindex
BOX_COLORS = [C["red"], C["green"], C["blue"], C["yellow"], C["pink"], C["lavender"],
              C["peach"], C["teal"], C["mauve"]]

FONT = "JetBrainsMono Nerd Font"
F_UI = (FONT, 10)
F_SM = (FONT, 9)
F_BOLD = (FONT, 10, "bold")


class YOLOLabelTool:
    def __init__(self, root, images_dir, labels_dir, classes):
        self.root = root
        self.root.title("YOLO Annotator")
        self.root.configure(bg=C["base"])
        self.root.geometry("1500x950")
        self.root.minsize(1100, 700)

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.classes = classes
        self.current_class = 0

        self.model = None
        self.model_path = None
        self.conf_threshold = 0.25

        def has_labels(img):
            lbl = self.labels_dir / (img.stem + ".txt")
            return lbl.exists() and lbl.stat().st_size > 0

        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ], key=lambda f: (has_labels(f), f.name))

        if not self.image_files:
            print("No images found!")
            root.quit()
            return

        # Fortschritt: welche Bilder sind schon gelabelt
        self.labeled_set = {f.stem for f in self.image_files if has_labels(f)}

        self.current_index = 0
        self.boxes = []          # [(class_id, x1, y1, x2, y2), ...] in Originalpixeln
        self.drawing = False
        self.start_x = self.start_y = None
        self.current_rect = None
        self.selected_box = None
        self.dirty = False       # ungespeicherte Änderungen

        self.setup_ui()
        self.load_image()

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        self._build_toolbar()

        body = Frame(self.root, bg=C["base"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_tool_rail(body)

        # Canvas-Bereich (zentriert auf dunklem Grund)
        canvas_wrap = Frame(body, bg=C["crust"])
        canvas_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = Canvas(canvas_wrap, cursor="none", bg=C["crust"],
                             highlightthickness=0, bd=0)
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")

        self._build_sidebar(body)
        self._build_statusbar()
        self._bind_events()

    def _build_toolbar(self):
        bar = Frame(self.root, bg=C["mantle"], height=52)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        Label(bar, text="  󰆦", font=(FONT, 16), bg=C["mantle"], fg=C["lavender"]).pack(side=tk.LEFT)
        Label(bar, text="YOLO Annotator", font=(FONT, 13, "bold"),
              bg=C["mantle"], fg=C["text"]).pack(side=tk.LEFT, padx=(2, 16))

        # Speicher-Status-Punkt + Bildname (mittig)
        self.dirty_dot = Label(bar, text="●", font=(FONT, 12), bg=C["mantle"], fg=C["green"])
        self.dirty_dot.pack(side=tk.LEFT, padx=(8, 4))
        self.name_label = Label(bar, text="", font=(FONT, 11), bg=C["mantle"], fg=C["subtext"])
        self.name_label.pack(side=tk.LEFT)

        # Fortschritt rechts
        self.progress_label = Label(bar, text="", font=F_SM, bg=C["mantle"], fg=C["subtext"])
        self.progress_label.pack(side=tk.RIGHT, padx=(8, 14))
        self.progress_canvas = Canvas(bar, width=160, height=8, bg=C["surface0"],
                                      highlightthickness=0, bd=0)
        self.progress_canvas.pack(side=tk.RIGHT, pady=22)

    def _mk_tool_btn(self, parent, glyph, cmd, tip, fg=None):
        btn = Label(parent, text=glyph, font=(FONT, 17), bg=C["crust"],
                    fg=fg or C["subtext"], width=3, pady=8, cursor="hand2")
        btn.pack(side=tk.TOP, pady=2)
        base_fg = fg or C["subtext"]
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>", lambda e: (btn.config(bg=C["surface0"], fg=C["text"]),
                                       self.set_status(tip)))
        btn.bind("<Leave>", lambda e: btn.config(bg=C["crust"], fg=base_fg))
        return btn

    def _build_tool_rail(self, parent):
        rail = Frame(parent, bg=C["crust"], width=52)
        rail.pack(side=tk.LEFT, fill=tk.Y)
        rail.pack_propagate(False)

        Frame(rail, height=6, bg=C["crust"]).pack()
        self._mk_tool_btn(rail, "󰁍", self.previous_image, "Vorheriges Bild  [A]")
        self._mk_tool_btn(rail, "󰁔", self.save_and_next, "Speichern + nächstes  [S]", fg=C["blue"])
        Frame(rail, height=1, bg=C["surface0"]).pack(fill=tk.X, padx=10, pady=6)
        self._mk_tool_btn(rail, "󰆓", self.save_labels, "Speichern")
        self._mk_tool_btn(rail, "󰚩", self.auto_label, "Auto-Label  [L]", fg=C["green"])
        self._mk_tool_btn(rail, "󰩺", self.delete_selected_or_last, "Box löschen  [D]", fg=C["red"])

    def _build_sidebar(self, parent):
        side = Frame(parent, bg=C["mantle"], width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # --- Klassen-Panel
        Label(side, text="  KLASSEN", font=(FONT, 9, "bold"), bg=C["mantle"],
              fg=C["overlay"], anchor="w").pack(fill=tk.X, pady=(12, 6))

        self.class_rows = []
        for i, name in enumerate(self.classes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            row = Frame(side, bg=C["mantle"], cursor="hand2")
            row.pack(fill=tk.X, padx=8, pady=1)
            swatch = Label(row, text="  ", bg=color, width=2)
            swatch.pack(side=tk.LEFT, padx=(6, 8), pady=5)
            name_lbl = Label(row, text=name, font=F_UI, bg=C["mantle"], fg=C["text"], anchor="w")
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            count_lbl = Label(row, text="0", font=F_SM, bg=C["mantle"], fg=C["overlay"], width=2)
            count_lbl.pack(side=tk.RIGHT, padx=(0, 6))
            key_lbl = Label(row, text=f"{i+1}" if i < 9 else "", font=F_SM,
                            bg=C["surface0"], fg=C["subtext"], width=3)
            key_lbl.pack(side=tk.RIGHT, padx=6)
            widgets = (row, swatch, name_lbl, count_lbl)
            for w in (row, name_lbl, swatch, count_lbl, key_lbl):
                w.bind("<Button-1>", lambda e, c=i: self.set_class(c))
            self.class_rows.append({"widgets": widgets, "name": name_lbl,
                                    "count": count_lbl, "color": color})

        # --- Annotationen-Panel
        Frame(side, height=1, bg=C["surface0"]).pack(fill=tk.X, padx=8, pady=(12, 0))
        head = Frame(side, bg=C["mantle"])
        head.pack(fill=tk.X, pady=(12, 6))
        Label(head, text="  ANNOTATIONEN", font=(FONT, 9, "bold"), bg=C["mantle"],
              fg=C["overlay"]).pack(side=tk.LEFT)
        self.annot_count = Label(head, text="0", font=F_SM, bg=C["mantle"], fg=C["subtext"])
        self.annot_count.pack(side=tk.RIGHT, padx=12)

        list_wrap = Frame(side, bg=C["mantle"])
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=8)
        scrollbar = Scrollbar(list_wrap, troughcolor=C["mantle"], bg=C["surface1"],
                              activebackground=C["overlay"], relief=tk.FLAT, bd=0, width=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.boxes_listbox = Listbox(
            list_wrap, yscrollcommand=scrollbar.set, font=F_SM,
            bg=C["crust"], fg=C["text"], bd=0, highlightthickness=0,
            selectbackground=C["surface1"], selectforeground=C["text"],
            activestyle="none")
        self.boxes_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.boxes_listbox.yview)
        self.boxes_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.boxes_listbox.bind("<Double-Button-1>", self.delete_selected_box)

        # --- Auto-Label-Panel
        Frame(side, height=1, bg=C["surface0"]).pack(fill=tk.X, padx=8, pady=(8, 0))
        Label(side, text="  KI-ASSISTENZ", font=(FONT, 9, "bold"), bg=C["mantle"],
              fg=C["overlay"], anchor="w").pack(fill=tk.X, pady=(10, 6))

        ai = Frame(side, bg=C["mantle"])
        ai.pack(fill=tk.X, padx=8)
        self.model_btn = Button(ai, text="󰆧  Modell wählen…", command=self.select_model,
                                font=F_SM, bg=C["surface0"], fg=C["text"],
                                activebackground=C["surface1"], activeforeground=C["text"],
                                relief=tk.FLAT, bd=0, pady=6, cursor="hand2")
        self.model_btn.pack(fill=tk.X)
        self.model_label = Label(side, text="kein Modell geladen", font=F_SM,
                                 bg=C["mantle"], fg=C["overlay"], anchor="w")
        self.model_label.pack(fill=tk.X, padx=12, pady=(4, 6))

        conf_row = Frame(side, bg=C["mantle"])
        conf_row.pack(fill=tk.X, padx=8)
        Label(conf_row, text="conf", font=F_SM, bg=C["mantle"], fg=C["subtext"]).pack(side=tk.LEFT)
        self.conf_scale = tk.Scale(
            conf_row, from_=0.05, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
            showvalue=True, font=(FONT, 8), bg=C["mantle"], fg=C["subtext"],
            troughcolor=C["surface0"], highlightthickness=0, bd=0,
            activebackground=C["lavender"], sliderrelief=tk.FLAT, length=180)
        self.conf_scale.set(self.conf_threshold)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.autolabel_button = Button(
            side, text="󰚩  Auto-Label  (L)", command=self.auto_label, state=tk.DISABLED,
            font=F_SM, bg=C["surface0"], fg=C["green"], disabledforeground=C["overlay"],
            activebackground=C["surface1"], activeforeground=C["green"],
            relief=tk.FLAT, bd=0, pady=7, cursor="hand2")
        self.autolabel_button.pack(fill=tk.X, padx=8, pady=(8, 12))

    def _build_statusbar(self):
        bar = Frame(self.root, bg=C["crust"], height=26)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        self.dims_label = Label(bar, text="", font=F_SM, bg=C["crust"], fg=C["overlay"])
        self.dims_label.pack(side=tk.LEFT, padx=12)
        self.coord_label = Label(bar, text="", font=F_SM, bg=C["crust"], fg=C["overlay"])
        self.coord_label.pack(side=tk.LEFT, padx=12)
        self.zoom_label = Label(bar, text="", font=F_SM, bg=C["crust"], fg=C["overlay"])
        self.zoom_label.pack(side=tk.LEFT, padx=12)
        self.status_label = Label(bar, text="bereit", font=F_SM, bg=C["crust"],
                                  fg=C["subtext"], anchor="e")
        self.status_label.pack(side=tk.RIGHT, padx=12)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: (self.canvas.delete("cross"),
                                               self.coord_label.config(text="")))

        self.root.bind("<Tab>", lambda e: self.cycle_class())
        for i in range(min(len(self.classes), 9)):
            self.root.bind(str(i + 1), lambda e, c=i: self.set_class(c))
        for k in ("d", "D"):
            self.root.bind(k, lambda e: self.delete_selected_or_last())
        for k in ("s", "S"):
            self.root.bind(k, lambda e: self.save_and_next())
        for k in ("a", "A"):
            self.root.bind(k, lambda e: self.previous_image())
        for k in ("l", "L"):
            self.root.bind(k, lambda e: self.auto_label())

    # -------------------------------------------------------------- Klassen

    def set_class(self, class_id):
        if 0 <= class_id < len(self.classes):
            self.current_class = class_id
            self.update_class_label()

    def cycle_class(self):
        self.set_class((self.current_class + 1) % len(self.classes))
        return "break"  # kein Tab-Fokuswechsel

    def update_class_label(self):
        counts = Counter(c for c, *_ in self.boxes)
        for i, row in enumerate(self.class_rows):
            active = (i == self.current_class)
            bg = C["surface1"] if active else C["mantle"]
            for w in row["widgets"]:
                w.config(bg=bg)
            row["name"].config(fg=C["text"] if active else C["subtext"],
                               font=F_BOLD if active else F_UI)
            n = counts.get(i, 0)
            row["count"].config(text=str(n),
                                fg=row["color"] if n else C["overlay"])

    def set_status(self, text, color=None):
        self.status_label.config(text=text, fg=color or C["subtext"])

    def set_dirty(self, dirty):
        self.dirty = dirty
        self.dirty_dot.config(fg=C["yellow"] if dirty else C["green"])

    # ---------------------------------------------------------- Auto-Label

    def select_model(self):
        path = filedialog.askopenfilename(
            title="YOLO-Modell wählen", initialdir=str(Path(__file__).parent),
            filetypes=[("YOLO weights", "*.pt"), ("Alle Dateien", "*")])
        if not path:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model_path = Path(path)
            self.model_label.config(text=f"󰄬 {self.model_path.name}", fg=C["green"])
            self.autolabel_button.config(state=tk.NORMAL)
            self.set_status(f"Modell geladen: {self.model_path.name}", C["green"])
            print(f"Modell geladen: {path} | Klassen: {self.model.names}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden:\n{e}")
            self.model = None
            self.model_label.config(text="kein Modell geladen", fg=C["overlay"])
            self.autolabel_button.config(state=tk.DISABLED)

    def auto_label(self):
        if self.model is None:
            messagebox.showinfo("Auto-Label", "Bitte zuerst ein Modell wählen.")
            return
        self.conf_threshold = float(self.conf_scale.get())
        img_path = self.image_files[self.current_index]
        self.set_status("Auto-Label läuft…", C["yellow"])
        self.root.update_idletasks()
        results = self.model.predict(str(img_path), conf=self.conf_threshold, verbose=False)

        added, skipped = 0, 0
        for r in results:
            for b in r.boxes:
                model_cls = int(b.cls[0])
                model_name = self.model.names.get(model_cls, str(model_cls))
                if model_name in self.classes:
                    class_id = self.classes.index(model_name)
                elif model_cls < len(self.classes):
                    class_id = model_cls
                    print(f"Warnung: Modellklasse '{model_name}' nicht in classes.txt, "
                          f"nutze Index {model_cls} ({self.classes[model_cls]})")
                else:
                    skipped += 1
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                self.boxes.append((class_id, x1, y1, x2, y2))
                added += 1

        self.set_dirty(True)
        self.draw_boxes()
        self.update_info()
        self.set_status(f"Auto-Label: +{added} Boxen, {skipped} übersprungen "
                        f"(conf ≥ {self.conf_threshold})",
                        C["green"] if added else C["yellow"])

    # -------------------------------------------------------- Bild / Labels

    def load_image(self):
        if not (0 <= self.current_index < len(self.image_files)):
            return
        img_path = self.image_files[self.current_index]
        self.current_image_name = img_path.name

        self.original_image = ImageOps.exif_transpose(Image.open(img_path))
        self.img_width, self.img_height = self.original_image.size

        max_w, max_h = 1280, 860
        ratio = min(max_w / self.img_width, max_h / self.img_height, 1.0)
        new_w, new_h = int(self.img_width * ratio), int(self.img_height * ratio)

        self.display_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.scale_x = self.img_width / new_w
        self.scale_y = self.img_height / new_h
        self.display_ratio = ratio

        self.selected_box = None
        self.set_dirty(False)
        self.load_labels()
        self.draw_boxes()
        self.update_info()
        self.update_class_label()

    def load_labels(self):
        self.boxes = []
        label_path = self.labels_dir / (Path(self.current_image_name).stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        xc = float(parts[1]) * self.img_width
                        yc = float(parts[2]) * self.img_height
                        w = float(parts[3]) * self.img_width
                        h = float(parts[4]) * self.img_height
                        self.boxes.append((class_id, xc - w/2, yc - h/2, xc + w/2, yc + h/2))

    def save_labels(self):
        label_path = self.labels_dir / (Path(self.current_image_name).stem + ".txt")
        with open(label_path, 'w') as f:
            for class_id, x1, y1, x2, y2 in self.boxes:
                xc = ((x1 + x2) / 2) / self.img_width
                yc = ((y1 + y2) / 2) / self.img_height
                w = (x2 - x1) / self.img_width
                h = (y2 - y1) / self.img_height
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        # Fortschritt aktualisieren
        stem = Path(self.current_image_name).stem
        if self.boxes:
            self.labeled_set.add(stem)
        else:
            self.labeled_set.discard(stem)
        self.set_dirty(False)
        self.update_info()
        self.set_status(f"gespeichert · {label_path.name}", C["green"])

    def draw_boxes(self):
        self.canvas.delete("box")
        self.canvas.delete("label")
        self.boxes_listbox.delete(0, tk.END)

        for i, (class_id, x1, y1, x2, y2) in enumerate(self.boxes):
            dx1, dy1 = x1 / self.scale_x, y1 / self.scale_y
            dx2, dy2 = x2 / self.scale_x, y2 / self.scale_y
            color = BOX_COLORS[class_id % len(BOX_COLORS)]
            sel = (i == self.selected_box)
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline=color,
                                         width=3 if sel else 2, tags="box")
            if sel:  # Eckmarker bei Auswahl
                for cx, cy in ((dx1, dy1), (dx2, dy1), (dx1, dy2), (dx2, dy2)):
                    self.canvas.create_rectangle(cx-3, cy-3, cx+3, cy+3,
                                                 fill=color, outline=C["crust"], tags="box")

            label_text = self.classes[class_id] if class_id < len(self.classes) else f"?{class_id}"
            text_id = self.canvas.create_text(dx1 + 6, dy1 + 3, text=label_text,
                                              fill=C["crust"], font=(FONT, 10, "bold"),
                                              anchor=tk.NW, stipple="gray25", tags="label")
            bb = self.canvas.bbox(text_id)
            bg_id = self.canvas.create_rectangle(bb[0]-4, bb[1]-2, bb[2]+4, bb[3]+2,
                                                 fill=color, outline="", stipple="gray25", tags="label")
            self.canvas.tag_raise(text_id, bg_id)

            self.boxes_listbox.insert(tk.END, f"  {label_text}   {int(x2-x1)}×{int(y2-y1)}")
            self.boxes_listbox.itemconfig(tk.END, foreground=color)

        if self.selected_box is not None and self.selected_box < len(self.boxes):
            self.boxes_listbox.selection_set(self.selected_box)
        self.annot_count.config(text=str(len(self.boxes)))
        self.update_class_label()

    # ------------------------------------------------------ Box-Löschlogik

    def find_box_at(self, img_x, img_y):
        hit, hit_area = None, None
        for i, (_, x1, y1, x2, y2) in enumerate(self.boxes):
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit_area is None or area < hit_area:
                    hit, hit_area = i, area
        return hit

    def on_right_click(self, event):
        index = self.find_box_at(event.x * self.scale_x, event.y * self.scale_y)
        if index is not None:
            self.delete_box(index)

    def on_listbox_select(self, event):
        selection = self.boxes_listbox.curselection()
        self.selected_box = selection[0] if selection else None
        self.draw_boxes()

    def delete_box(self, index):
        if 0 <= index < len(self.boxes):
            self.boxes.pop(index)
            if self.selected_box is not None:
                if self.selected_box == index:
                    self.selected_box = None
                elif self.selected_box > index:
                    self.selected_box -= 1
            self.set_dirty(True)
            self.draw_boxes()
            self.update_info()

    def delete_selected_or_last(self):
        if self.selected_box is not None:
            self.delete_box(self.selected_box)
        elif self.boxes:
            self.delete_box(len(self.boxes) - 1)

    def delete_selected_box(self, event):
        selection = self.boxes_listbox.curselection()
        if selection:
            self.delete_box(selection[0])

    # -------------------------------------------------- Zeichnen / Navigation

    def _guide_line(self, x1, y1, x2, y2):
        # zweilagige Linie: dunkler Halo + helle Linie -> auf jedem Untergrund sichtbar
        self.canvas.create_line(x1, y1, x2, y2, fill=C["crust"], width=3, tags="cross")
        self.canvas.create_line(x1, y1, x2, y2, fill="#eef1ff", width=1, tags="cross")

    def on_mouse_move(self, event):
        # Fadenkreuz-Führung wie in professionellen Tools (mit Lücke + Klassenmarker)
        self.canvas.delete("cross")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = event.x, event.y
        gap = 7
        self._guide_line(x, 0, x, y - gap)
        self._guide_line(x, y + gap, x, h)
        self._guide_line(0, y, x - gap, y)
        self._guide_line(x + gap, y, w, y)
        col = BOX_COLORS[self.current_class % len(BOX_COLORS)]
        self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                     outline=col, width=2, tags="cross")
        self.canvas.create_line(x, y - 3, x, y + 3, fill=col, tags="cross")
        self.canvas.create_line(x - 3, y, x + 3, y, fill=col, tags="cross")
        self.coord_label.config(text=f"󰍽 {int(event.x*self.scale_x)}, {int(event.y*self.scale_y)}")

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x * self.scale_x
        self.start_y = event.y * self.scale_y

    def on_mouse_drag(self, event):
        if not self.drawing:
            return
        self.canvas.delete("cross")
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        color = BOX_COLORS[self.current_class % len(BOX_COLORS)]
        self.current_rect = self.canvas.create_rectangle(
            self.start_x / self.scale_x, self.start_y / self.scale_y,
            event.x, event.y, outline=color, width=2, dash=(4, 4))

    def on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        end_x, end_y = event.x * self.scale_x, event.y * self.scale_y
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return
        self.boxes.append((self.current_class, x1, y1, x2, y2))
        self.selected_box = len(self.boxes) - 1
        self.set_dirty(True)
        self.draw_boxes()
        self.update_info()

    def save_and_next(self):
        self.save_labels()
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image()
        else:
            self.current_index = len(self.image_files) - 1
            self.set_status("alle Bilder erreicht 󰄬", C["green"])

    def previous_image(self):
        self.save_labels()
        self.current_index = max(0, self.current_index - 1)
        self.load_image()

    def update_info(self):
        total = len(self.image_files)
        current = self.current_index + 1
        self.name_label.config(text=self.current_image_name)
        done = len(self.labeled_set)
        self.progress_label.config(text=f"{done}/{total} gelabelt  ·  Bild {current}")
        self.dims_label.config(text=f"󰋩 {self.img_width}×{self.img_height}")
        self.zoom_label.config(text=f"󰍉 {int(self.display_ratio*100)}%")
        self.annot_count.config(text=str(len(self.boxes)))
        # Fortschrittsbalken
        self.progress_canvas.delete("all")
        frac = done / total if total else 0
        self.progress_canvas.create_rectangle(0, 0, int(160 * frac), 8,
                                               fill=C["green"], outline="")


def load_classes(classes_file="classes.txt"):
    """Liest die Klassen aus classes.txt (eine Klasse pro Zeile)."""
    path = Path(classes_file)
    if path.exists():
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return ["dynamic_buoy", "outer_buoy"]


def main():
    root = tk.Tk()
    images_dir = "data/images"
    labels_dir = "data/labels"
    classes = load_classes()
    app = YOLOLabelTool(root, images_dir, labels_dir, classes)
    root.mainloop()


if __name__ == "__main__":
    main()
