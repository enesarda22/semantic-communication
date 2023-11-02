import os
import csv


class folder_reader:
    def _init_(self, path):
        self.path = path
        self.text = []

    def read_folder(self):
        os.chdir(self.path)
        counter = 0
        for file in os.listdir():
            print("Iter: " + str(counter))
            counter += 1
            if file.endswith(".txt"):
                file_path = f"{self.path}\{file}"
                self.text.append("".join(self.read_text_file(file_path)))

    def read_text_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read()

    def save(self):

        with open("processed.csv", mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the strings to the CSV file
            for string in self.text:
                writer.writerow([string])