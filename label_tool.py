"""
Simple YOLO Labeling Tool
- Draw bounding boxes by clicking and dragging
- Press Tab to cycle through classes (number keys also work)
- Press 'd' to delete the selected box (or the last one)
- Right-click on a box to delete it
- Press 's' to save and go to next image
- Press 'a' to go to previous image
- Auto-label: pick a .pt model and let it pre-label the current image
"""

import tkinter as tk
from tkinter import Canvas, Label, Frame, Listbox, Scrollbar, Button, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
from pathlib import Path

# Catppuccin Mocha
C = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
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

# Boxfarben je Klasse: buoy, yellow-buoy, ship, parking-buoy, buoy_new, big_buoy
BOX_COLORS = [C["red"], C["green"], C["blue"], C["yellow"], C["pink"], C["lavender"],
              C["peach"], C["teal"], C["mauve"]]

FONT = "JetBrainsMono Nerd Font"


class YOLOLabelTool:
    def __init__(self, root, images_dir, labels_dir, classes):
        self.root = root
        self.root.title("YOLO Label Tool — Monaco Energy Boat Challenge")
        self.root.configure(bg=C["base"])

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.classes = classes
        self.current_class = 0

        self.model = None
        self.model_path = None
        self.conf_threshold = 0.25

        # Get all images — ungelabelte zuerst, danach alphabetisch
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

        self.current_index = 0
        self.boxes = []  # [(class_id, x1, y1, x2, y2), ...]
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selected_box = None

        self.setup_ui()
        self.load_image()

    # --- UI ---

    def setup_ui(self):
        # Header: Bildinfo links, Shortcuts rechts
        header = Frame(self.root, bg=C["mantle"])
        header.pack(side=tk.TOP, fill=tk.X)

        self.info_label = Label(header, text="", font=(FONT, 12, "bold"),
                                bg=C["mantle"], fg=C["text"], padx=12, pady=8)
        self.info_label.pack(side=tk.LEFT)

        self.progress_label = Label(header, text="", font=(FONT, 10),
                                    bg=C["mantle"], fg=C["subtext"])
        self.progress_label.pack(side=tk.LEFT, padx=8)

        help_text = ("󰌌  Tab: Klasse   󰍽  Ziehen: Box   󰍴  Rechtsklick: Löschen   "
                     "D: Löschen   S: Speichern+Weiter   A: Zurück   L: Auto-Label")
        Label(header, text=help_text, font=(FONT, 9),
              bg=C["mantle"], fg=C["overlay"], padx=12).pack(side=tk.RIGHT)

        # Klassen-Leiste: klickbare Chips mit Boxfarbe
        class_bar = Frame(self.root, bg=C["base"])
        class_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        Label(class_bar, text="Klassen", font=(FONT, 10, "bold"),
              bg=C["base"], fg=C["subtext"]).pack(side=tk.LEFT, padx=(0, 10))

        self.class_buttons = []
        for i, name in enumerate(self.classes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            btn = Label(class_bar, text=f" {i + 1}  {name} ", font=(FONT, 10),
                        bg=C["surface0"], fg=color, padx=8, pady=4, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=3)
            btn.bind("<Button-1>", lambda e, c=i: self.set_class(c))
            self.class_buttons.append(btn)

        # Mitte: Canvas links, Boxliste rechts
        middle_frame = Frame(self.root, bg=C["base"])
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.canvas = Canvas(middle_frame, cursor="cross", bg=C["crust"],
                             highlightthickness=1, highlightbackground=C["surface1"])
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_frame = Frame(middle_frame, bg=C["base"])
        list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        Label(list_frame, text="Boxen", font=(FONT, 10, "bold"),
              bg=C["base"], fg=C["subtext"], anchor="w").pack(fill=tk.X, pady=(0, 4))

        scrollbar = Scrollbar(list_frame, troughcolor=C["mantle"], bg=C["surface1"],
                              activebackground=C["overlay"], relief=tk.FLAT, bd=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.boxes_listbox = Listbox(
            list_frame, width=32, yscrollcommand=scrollbar.set, font=(FONT, 9),
            bg=C["mantle"], fg=C["text"], bd=0, highlightthickness=1,
            highlightbackground=C["surface1"], highlightcolor=C["lavender"],
            selectbackground=C["surface1"], selectforeground=C["lavender"],
            activestyle="none")
        self.boxes_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.boxes_listbox.yview)

        # Fußleiste: Auto-Label-Werkzeuge + Statuszeile
        footer = Frame(self.root, bg=C["mantle"])
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        btn_style = dict(font=(FONT, 10), bg=C["surface0"], fg=C["text"],
                         activebackground=C["surface1"], activeforeground=C["text"],
                         relief=tk.FLAT, bd=0, padx=12, pady=5, cursor="hand2")

        Button(footer, text="󰆧  Modell wählen…", command=self.select_model,
               **btn_style).pack(side=tk.LEFT, padx=(12, 6), pady=8)

        self.model_label = Label(footer, text="kein Modell", font=(FONT, 9),
                                 bg=C["mantle"], fg=C["overlay"])
        self.model_label.pack(side=tk.LEFT, padx=4)

        self.autolabel_button = Button(footer, text="󰚩  Auto-Label (L)",
                                       command=self.auto_label, state=tk.DISABLED,
                                       **{**btn_style, "fg": C["green"],
                                          "disabledforeground": C["overlay"]})
        self.autolabel_button.pack(side=tk.LEFT, padx=10, pady=8)

        Label(footer, text="conf", font=(FONT, 9),
              bg=C["mantle"], fg=C["subtext"]).pack(side=tk.LEFT, padx=(10, 2))
        self.conf_scale = tk.Scale(
            footer, from_=0.05, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
            length=140, showvalue=True, font=(FONT, 8),
            bg=C["mantle"], fg=C["text"], troughcolor=C["surface0"],
            highlightthickness=0, bd=0, activebackground=C["lavender"])
        self.conf_scale.set(self.conf_threshold)
        self.conf_scale.pack(side=tk.LEFT)

        self.status_label = Label(footer, text="", font=(FONT, 9),
                                  bg=C["mantle"], fg=C["subtext"], padx=12)
        self.status_label.pack(side=tk.RIGHT)

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.root.bind("<Tab>", lambda e: self.cycle_class())
        for i in range(min(len(self.classes), 9)):
            self.root.bind(str(i + 1), lambda e, c=i: self.set_class(c))
        self.root.bind("d", lambda e: self.delete_selected_or_last())
        self.root.bind("D", lambda e: self.delete_selected_or_last())
        self.root.bind("s", lambda e: self.save_and_next())
        self.root.bind("S", lambda e: self.save_and_next())
        self.root.bind("a", lambda e: self.previous_image())
        self.root.bind("A", lambda e: self.previous_image())
        self.root.bind("l", lambda e: self.auto_label())
        self.root.bind("L", lambda e: self.auto_label())

        self.boxes_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.boxes_listbox.bind("<Double-Button-1>", self.delete_selected_box)

    def set_class(self, class_id):
        if 0 <= class_id < len(self.classes):
            self.current_class = class_id
            self.update_class_label()

    def cycle_class(self):
        self.set_class((self.current_class + 1) % len(self.classes))
        return "break"  # suppress default tkinter Tab focus traversal

    def update_class_label(self):
        for i, btn in enumerate(self.class_buttons):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            if i == self.current_class:
                btn.config(bg=color, fg=C["crust"], font=(FONT, 10, "bold"))
            else:
                btn.config(bg=C["surface0"], fg=color, font=(FONT, 10))

    def set_status(self, text, color=None):
        self.status_label.config(text=text, fg=color or C["subtext"])

    # --- Auto-Labeling ---

    def select_model(self):
        path = filedialog.askopenfilename(
            title="YOLO-Modell wählen",
            initialdir=str(Path(__file__).parent),
            filetypes=[("YOLO weights", "*.pt"), ("Alle Dateien", "*")]
        )
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
            self.model_label.config(text="kein Modell", fg=C["overlay"])
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

        # Modellklassen auf die Datensatz-Klassen mappen: gleicher Name gewinnt,
        # sonst gleicher Index (mit Warnung), unbekannte Klassen werden übersprungen.
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

        self.draw_boxes()
        self.update_info()
        self.set_status(f"Auto-Label: +{added} Boxen, {skipped} übersprungen "
                        f"(conf ≥ {self.conf_threshold})",
                        C["green"] if added else C["yellow"])
        print(f"Auto-Label: {added} Boxen hinzugefügt, {skipped} übersprungen "
              f"({img_path.name}, conf>={self.conf_threshold})")

    # --- Image / Labels ---

    def load_image(self):
        if not (0 <= self.current_index < len(self.image_files)):
            return

        img_path = self.image_files[self.current_index]
        self.current_image_name = img_path.name

        # Load image, applying EXIF orientation (many photos carry orientation tag 6)
        self.original_image = ImageOps.exif_transpose(Image.open(img_path))
        self.img_width, self.img_height = self.original_image.size

        # Resize for display (max 1200x800)
        max_w, max_h = 1200, 800
        ratio = min(max_w / self.img_width, max_h / self.img_height, 1.0)
        new_w = int(self.img_width * ratio)
        new_h = int(self.img_height * ratio)

        self.display_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.scale_x = self.img_width / new_w
        self.scale_y = self.img_height / new_h

        self.selected_box = None

        # Load existing labels
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
                        x_center = float(parts[1]) * self.img_width
                        y_center = float(parts[2]) * self.img_height
                        width = float(parts[3]) * self.img_width
                        height = float(parts[4]) * self.img_height

                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        self.boxes.append((class_id, x1, y1, x2, y2))

    def save_labels(self):
        label_path = self.labels_dir / (Path(self.current_image_name).stem + ".txt")

        with open(label_path, 'w') as f:
            for class_id, x1, y1, x2, y2 in self.boxes:
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / self.img_width
                y_center = ((y1 + y2) / 2) / self.img_height
                width = (x2 - x1) / self.img_width
                height = (y2 - y1) / self.img_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        self.set_status(f"Gespeichert: {label_path.name}", C["green"])
        print(f"Saved: {label_path}")

    def draw_boxes(self):
        # Clear existing box drawings
        self.canvas.delete("box")
        self.canvas.delete("label")

        # Update listbox
        self.boxes_listbox.delete(0, tk.END)

        # Draw all boxes
        for i, (class_id, x1, y1, x2, y2) in enumerate(self.boxes):
            # Scale to display coordinates
            dx1 = x1 / self.scale_x
            dy1 = y1 / self.scale_y
            dx2 = x2 / self.scale_x
            dy2 = y2 / self.scale_y

            color = BOX_COLORS[class_id % len(BOX_COLORS)]
            width = 4 if i == self.selected_box else 2
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                        outline=color, width=width, tags="box")

            # Draw class label on a filled tag for readability
            label_text = self.classes[class_id] if class_id < len(self.classes) else f"?{class_id}"
            text_id = self.canvas.create_text(dx1 + 6, dy1 + 4,
                                              text=label_text,
                                              fill=C["crust"],
                                              font=(FONT, 10, "bold"),
                                              anchor=tk.NW,
                                              stipple="gray25",
                                              tags="label")
            bbox = self.canvas.bbox(text_id)
            bg_id = self.canvas.create_rectangle(bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2,
                                                 fill=color, outline=color, stipple="gray25", tags="label")
            self.canvas.tag_raise(text_id, bg_id)

            # Add to listbox
            self.boxes_listbox.insert(tk.END, f"{i+1:>2}  {label_text}  ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            self.boxes_listbox.itemconfig(tk.END, foreground=color)

        if self.selected_box is not None and self.selected_box < len(self.boxes):
            self.boxes_listbox.selection_set(self.selected_box)

    # --- Box deletion logic ---

    def find_box_at(self, img_x, img_y):
        """Return the index of the smallest box containing the point, or None."""
        hit, hit_area = None, None
        for i, (_, x1, y1, x2, y2) in enumerate(self.boxes):
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit_area is None or area < hit_area:
                    hit, hit_area = i, area
        return hit

    def on_right_click(self, event):
        img_x = event.x * self.scale_x
        img_y = event.y * self.scale_y
        index = self.find_box_at(img_x, img_y)
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

    # --- Drawing / Navigation ---

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x * self.scale_x
        self.start_y = event.y * self.scale_y

    def on_mouse_drag(self, event):
        if not self.drawing:
            return

        # Remove previous rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)

        # Draw new rectangle
        end_x = event.x
        end_y = event.y

        color = BOX_COLORS[self.current_class % len(BOX_COLORS)]

        self.current_rect = self.canvas.create_rectangle(
            self.start_x / self.scale_x,
            self.start_y / self.scale_y,
            end_x, end_y,
            outline=color, width=2, dash=(4, 4)
        )

    def on_mouse_up(self, event):
        if not self.drawing:
            return

        self.drawing = False

        end_x = event.x * self.scale_x
        end_y = event.y * self.scale_y

        # Remove drawing rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        # Add box (ensure x1 < x2, y1 < y2)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        # Ignore very small boxes
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return

        self.boxes.append((self.current_class, x1, y1, x2, y2))
        self.draw_boxes()
        self.update_info()

    def save_and_next(self):
        self.save_labels()
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image()
        else:
            print("All images labeled!")
            self.set_status("Alle Bilder gelabelt! 󰄬", C["green"])
            self.update_info()

    def previous_image(self):
        self.save_labels()
        self.current_index -= 1
        if self.current_index >= 0:
            self.load_image()
        else:
            self.current_index = 0

    def update_info(self):
        total = len(self.image_files)
        current = self.current_index + 1 if self.current_index < total else total
        self.info_label.config(text=self.current_image_name)
        self.progress_label.config(
            text=f"{current}/{total}  ·  {len(self.boxes)} Boxen")


def load_classes(classes_file="classes.txt"):
    """Liest die Klassen aus classes.txt (eine Klasse pro Zeile)."""
    path = Path(classes_file)
    if path.exists():
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback, falls keine classes.txt vorhanden ist
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
