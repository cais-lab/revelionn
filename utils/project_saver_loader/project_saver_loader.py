import os
import sys
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(cur_path))
sys.path.append(root_path)

try:
    from utils.project_saver_loader.save_load_project import save, load
except ModuleNotFoundError:
    raise


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Project Saver/Loader')
        self.geometry('300x140')
        btn_file = tk.Button(self, text="Save the project",
                             command=self.save_project, font=16, width=30)
        btn_dir = tk.Button(self, text="Load a project",
                            command=self.load_project, font=16, width=30)
        btn_file.pack(padx=60, pady=15)
        btn_dir.pack(padx=60, pady=15)

    @staticmethod
    def save_project():
        new_file = fd.asksaveasfilename(title="Save the project", defaultextension=".rvlcp",
                                        filetypes=(("RevelioNN project file", "*.rvlcp"),))
        if new_file:
            save(new_file)
            msg = 'Project successfully saved'
            mb.showinfo("Information", msg)

    @staticmethod
    def load_project():
        filename = fd.askopenfilename(title="Load a project", initialdir="/",
                                      filetypes=(("RevelioNN project file", "*.rvlcp"),))
        if filename:
            load(filename)
            msg = 'Project loaded successfully'
            mb.showinfo("Information", msg)


if __name__ == "__main__":
    app = App()
    app.mainloop()
